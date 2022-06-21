####################################################
# Adapted from https://github.com/Rudrabha/Wav2Lip #
####################################################
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.model import SyncTransformer
from sklearn.metrics import f1_score
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data as data_utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchaudio.transforms import MelScale
from glob import glob
import acappella_info
import os, random, cv2, argparse
from hparams import hparams
from natsort import natsorted
import soundfile as sf
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator on the Acappella dataset')
parser.add_argument("--data_root", help="Root folder of the preprocessed Acappella dataset",
                    default="/mnt/DATA/dataset/acapsol/acappella/")
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',
                    default=None,
                    type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory',
                    default='/mnt/DATA/dataset/acapsol/experiments/vocalist_5f_acappella',
                    type=str)

args = parser.parse_args()
writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'tensorboard'))

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

# For the context of 5 visual frames, num_audio_elements = 16000 * (5/25) = 3200,
num_audio_elements = 3200  # 6400  # 16000/25 * v_context
tot_num_frames = 250  # buffer
v_context = 5  # 10  # 25
BATCH_SIZE = 128
MODE = 'train'
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, f_max=hparams.fmax,
                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney').to(0)

logloss = nn.BCEWithLogitsLoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d, y)

    return d, loss


class Dataset(object):
    def __init__(self, split):
        self.split = split
        self.all_videos = natsorted(list(glob(os.path.join(args.data_root, 'jpgs', split, '*/*/*'))),
                                    key=lambda y: y.lower())
        self.all_audios = natsorted(list(glob(os.path.join(args.data_root, 'splits_16k', split, 'audio', '*/*/*.wav'))),
                                    key=lambda y: y.lower())
        self.ts = acappella_info.get_timestamps()

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_wav(self, wavpath, vid_frame_id):
        aud = sf.SoundFile(wavpath)
        can_seek = aud.seekable()
        pos_aud_chunk_start = vid_frame_id * 640
        _ = aud.seek(pos_aud_chunk_start)
        wav_vec = aud.read(num_audio_elements)
        return wav_vec

    def rms(self, x):
        val = np.sqrt(np.mean(x ** 2))
        if val == 0:
            val = 1
        return val

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + v_context):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        if self.split == 'train':
            return BATCH_SIZE * 359  # len(self.all_videos)
        else:
            return BATCH_SIZE * 50

    def __getitem__(self, idx):
        while 1:
            try:
                idx = random.randint(0, len(self.all_videos) - 1)
                vidname = self.all_videos[idx]
                wavpath = self.all_audios[idx]
                sample_id = basename(vidname)
                interval = random.choice(self.ts[sample_id])
                interval_st, interval_end = interval[0], interval[1]
                if interval_end - interval_st <= tot_num_frames:
                    continue
                pos_frame_id = random.randint(interval_st, interval_end - v_context)
                pos_wav = self.get_wav(wavpath, pos_frame_id)
                rms_pos_wav = self.rms(pos_wav)

                img_name = os.path.join(vidname, str(pos_frame_id) + '.jpg')
                window_fnames = self.get_window(img_name)
                if window_fnames is None:
                    continue

                window = []
                all_read = True
                for fname in window_fnames:
                    img = cv2.imread(fname)
                    if img is None:
                        all_read = False
                        break
                    try:
                        img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                    except Exception as e:
                        all_read = False
                        break

                    window.append(img)

                if not all_read: continue
                if random.choice([True, False]):
                    y = torch.ones(1).float()
                    wav = pos_wav
                else:
                    y = torch.zeros(1).float()
                    try_counter = 0
                    while (True):
                        neg_frame_id = random.randint(interval_st, interval_end - v_context)
                        if neg_frame_id != pos_frame_id:
                            wav = self.get_wav(wavpath, neg_frame_id)
                            if rms_pos_wav > 0.01:
                                break
                            else:
                                if self.rms(wav) > 0.01 or try_counter > 10:
                                    break
                            try_counter += 1

                    if try_counter > 10:
                        continue
                aud_tensor = torch.FloatTensor(wav)

                # H, W, T, 3 --> T*3
                vid = np.concatenate(window, axis=2) / 255.
                vid = vid.transpose(2, 0, 1)
                vid = torch.FloatTensor(vid[:, 48:])
            except Exception as e:
                continue
            return vid, aud_tensor, y


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    while global_epoch < nepochs:
        f1_scores = []
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (vid, aud, y) in prog_bar:
            vid = vid.to(device)
            gt_aud = aud.to(device)

            spec = torch.stft(gt_aud, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size).to(gt_aud.device), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            # NORMALIZED MEL
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            mels = normalized_mel[:, :, :-1].unsqueeze(1)
            model.train()
            optimizer.zero_grad()

            out = model(vid.clone().detach(), mels.clone().detach())
            loss = logloss(out, y.squeeze(-1).to(device))
            loss.backward()
            optimizer.step()

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(),
                                 est_label.clone().detach().cpu().numpy(),
                                 average="weighted")
            f1_scores.append(f1_metric.item())
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            prog_bar.set_description('[TRAINING LOSS]: {}, [TRAINING F1]: {}'
                                     .format(running_loss / (step + 1), sum(f1_scores) / len(f1_scores)))

        f1_epoch = sum(f1_scores) / len(f1_scores)
        writer.add_scalars('f1_epoch', {'train': f1_epoch}, global_epoch)
        writer.add_scalars('loss_epoch', {'train': running_loss / (step + 1)}, global_epoch)

        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

        with torch.no_grad():
            eval_model(test_data_loader, device, model, checkpoint_dir)

        global_epoch += 1


def eval_model(test_data_loader, device, model, checkpoint_dir, nepochs=None):
    losses = []
    running_loss = 0
    f1_scores = []
    prog_bar = tqdm(enumerate(test_data_loader))
    for step, (vid, aud, y) in prog_bar:
        model.eval()
        with torch.no_grad():
            vid = vid.to(device)
            gt_aud = aud.to(device)
            mini_batch_size = vid.shape[0]

            spec = torch.stft(gt_aud, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size).to(gt_aud.device), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            # NORMALIZED MEL
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            mels = normalized_mel[:, :, :-1].unsqueeze(1)
            out = model(vid.clone().detach(), mels.clone().detach())
            loss = logloss(out, y.squeeze(-1).to(device))
            losses.append(loss.item())

            est_label = (out > 0.5).float()
            f1_metric = f1_score(y.clone().detach().cpu().numpy(),
                                 est_label.clone().detach().cpu().numpy(),
                                 average="weighted")
            f1_scores.append(f1_metric.item())
            running_loss += loss.item()
            prog_bar.set_description('[VAL RUNNING LOSS]: {}, [VAL F1]: {}'
                                     .format(running_loss / (step + 1), sum(f1_scores) / len(f1_scores)))

    averaged_loss = sum(losses) / len(losses)
    writer.add_scalars('loss_epoch', {'val': averaged_loss}, global_epoch)
    writer.add_scalars('f1_epoch', {'val': sum(f1_scores) / len(f1_scores)}, global_epoch)
    return


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val_seen')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=24)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=24)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncTransformer().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=5e-5)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=100,
          nepochs=1000)
