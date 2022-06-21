##########################################################################################
# Adapted from: https://github.com/joonson/syncnet_python/blob/master/SyncNetInstance.py #
##########################################################################################
from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from models.model import SyncTransformer
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils import data as data_utils
import numpy as np
from torchaudio.transforms import MelScale
from glob import glob
import acappella_info
import os, cv2, argparse
from hparams import hparams
from natsort import natsorted
import soundfile as sf
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/mnt/DATA/dataset/acapsol/acappella/")
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',
                    default="/mnt/DATA/dataset/acapsol/experiments/vocalist_weights/vocalist_5f_acappella.pth",
                    # default="/mnt/DATA/dataset/acapsol/experiments/vocalist_weights/vocalist_5f_lrs2.pth",
                    type=str)

args = parser.parse_args()


use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

v_context = 5
mel_step_size = 16  # num_audio_elements/hop_size
BATCH_SIZE = 1
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, f_min=hparams.fmin, f_max=hparams.fmax,
                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney')
logloss = nn.BCEWithLogitsLoss()


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

    def get_wav(self, wavpath):
        return sf.read(wavpath)[0]

    def get_window(self, start_frame, end):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, end):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            vidname = self.all_videos[idx]
            wavpath = self.all_audios[idx]
            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            wav = self.get_wav(wavpath)
            min_length = min(len(img_names), math.floor(len(wav) / 640))
            lastframe = min_length - 5

            img_name = os.path.join(vidname, '0.jpg')
            window_fnames = self.get_window(img_name, len(img_names))
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
            # H, W, T, 3 --> T*3
            vid = np.concatenate(window, axis=2) / 255.
            vid = vid.transpose(2, 0, 1)
            vid = torch.FloatTensor(vid[:, 48:])

            aud_tensor = torch.FloatTensor(wav)
            spec = torch.stft(aud_tensor, n_fft=hparams.n_fft, hop_length=hparams.hop_size, win_length=hparams.win_size,
                              window=torch.hann_window(hparams.win_size), return_complex=True)
            melspec = melscale(torch.abs(spec.detach().clone()).float())
            melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
            # NORMALIZED MEL
            normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                        -hparams.max_abs_value, hparams.max_abs_value)
            mels = normalized_mel.unsqueeze(0)
            if torch.any(torch.isnan(vid)) or torch.any(torch.isnan(mels)):
                continue
            if vid == None or mels == None:
                continue
            return vid, mels, lastframe


def calc_pdist(model, feat1, feat2, vshift=15):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2.permute(1, 2, 3, 0).contiguous(), (vshift, vshift)).permute(3, 0, 1,
                                                                                                       2).contiguous()

    dists = []
    num_rows_dist = len(feat1)
    for i in range(0, num_rows_dist):

        raw_sync_scores = model(feat1[i].unsqueeze(0).repeat(win_size, 1, 1, 1).to(device),
                                feat2p[i:i + win_size, :].to(device))
        dist_measures = raw_sync_scores.clone().cpu()
        if i in range(vshift):
            dist_measures[0:vshift - i] = torch.tensor(-1000, dtype=torch.float).to(device)
        elif i in range(num_rows_dist - vshift, num_rows_dist):
            dist_measures[vshift + num_rows_dist - i:] = torch.tensor(-1000, dtype=torch.float).to(device)

        dists.append(dist_measures)

    return dists


def eval_model(test_data_loader, device, model):
    prog_bar = tqdm(enumerate(test_data_loader))
    samplewise_acc_k5_t1, samplewise_acc_k5_t5, samplewise_acc_k10_t1, samplewise_acc_k10_t5, samplewise_acc_k15_t1, samplewise_acc_k15_t5, samplewise_acc_k20_t1, samplewise_acc_k20_t5, samplewise_acc_k25_t1, samplewise_acc_k25_t5 = [], [], [], [], [], [], [], [], [], []
    for step, (vid, aud, lastframe) in prog_bar:
        model.eval()
        with torch.no_grad():
            vid = vid.view(BATCH_SIZE, (lastframe + v_context), 3, 48, 96)
            batch_size = 20
            lastframe = lastframe.item()
            lim_in = []
            lcc_in = []
            for i in range(0, lastframe, batch_size):
                im_batch = [vid[:, vframe:vframe + v_context, :, :, :].view(BATCH_SIZE, -1, 48, 96) for vframe in
                            range(i, min(lastframe, i + batch_size))]
                im_in = torch.cat(im_batch, 0)
                lim_in.append(im_in)

                cc_batch = [
                    aud[:, :, :, int(80. * (vframe / float(hparams.fps))):int(80. * (vframe / float(hparams.fps))) + mel_step_size]
                    for vframe in
                    range(i, min(lastframe, i + batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                lcc_in.append(cc_in)

            lim_in = torch.cat(lim_in, 0)
            lcc_in = torch.cat(lcc_in, 0)
            dist = calc_pdist(model, lim_in, lcc_in, vshift=hparams.v_shift)

            # K=5
            dist_tensor_k5 = torch.stack(dist)
            offsets_k5 = hparams.v_shift - torch.argmax(dist_tensor_k5, dim=1)
            cur_num_correct_pred_k5_t1 = len(torch.where(abs(offsets_k5) <= 1)[0])
            cur_num_correct_pred_k5_t5 = len(torch.where(abs(offsets_k5) <= 5)[0])
            samplewise_acc_k5_t1.append(cur_num_correct_pred_k5_t1 / len(offsets_k5))
            samplewise_acc_k5_t5.append(cur_num_correct_pred_k5_t5 / len(offsets_k5))

            # K=10
            dist_tensor_k10 = (dist_tensor_k5[3:-2] + dist_tensor_k5[2:-3] + dist_tensor_k5[4:-1]
                               + dist_tensor_k5[1:-4] + dist_tensor_k5[5:] + dist_tensor_k5[:-5]) / 6
            dk10_m1 = torch.mean(dist_tensor_k5[:5], dim=0).unsqueeze(0)
            dk10_p1 = torch.mean(dist_tensor_k5[-5:], dim=0).unsqueeze(0)
            dk10_m2 = torch.mean(dist_tensor_k5[:4], dim=0).unsqueeze(0)
            dk10_p2 = torch.mean(dist_tensor_k5[-4:], dim=0).unsqueeze(0)
            dk10_m3 = torch.mean(dist_tensor_k5[:3], dim=0).unsqueeze(0)
            dist_tensor_k10 = torch.cat([dk10_m3, dk10_m2, dk10_m1, dist_tensor_k10, dk10_p1, dk10_p2], dim=0)
            offsets_k10 = hparams.v_shift - torch.argmax(dist_tensor_k10, dim=1)
            cur_num_correct_pred_k10_t1 = len(torch.where(abs(offsets_k10) <= 1)[0])
            cur_num_correct_pred_k10_t5 = len(torch.where(abs(offsets_k10) <= 5)[0])
            samplewise_acc_k10_t1.append(cur_num_correct_pred_k10_t1 / len(offsets_k10))
            samplewise_acc_k10_t5.append(cur_num_correct_pred_k10_t5 / len(offsets_k10))

            # K=15
            dist_tensor_k15 = (dist_tensor_k5[5:-5] + dist_tensor_k5[4:-6] + dist_tensor_k5[6:-4] +
                               dist_tensor_k5[3:-7] + dist_tensor_k5[7:-3] + dist_tensor_k5[2:-8] +
                               dist_tensor_k5[8:-2] + dist_tensor_k5[1:-9] + dist_tensor_k5[9:-1] +
                               dist_tensor_k5[:-10] + dist_tensor_k5[10:]) / 11
            dk15_m1 = torch.mean(dist_tensor_k5[:10], dim=0).unsqueeze(0)
            dk15_p1 = torch.mean(dist_tensor_k5[-10:], dim=0).unsqueeze(0)
            dk15_m2 = torch.mean(dist_tensor_k5[:9], dim=0).unsqueeze(0)
            dk15_p2 = torch.mean(dist_tensor_k5[-9:], dim=0).unsqueeze(0)
            dk15_m3 = torch.mean(dist_tensor_k5[:8], dim=0).unsqueeze(0)
            dk15_p3 = torch.mean(dist_tensor_k5[-8:], dim=0).unsqueeze(0)
            dk15_m4 = torch.mean(dist_tensor_k5[:7], dim=0).unsqueeze(0)
            dk15_p4 = torch.mean(dist_tensor_k5[-7:], dim=0).unsqueeze(0)
            dk15_m5 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk15_p5 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)

            dist_tensor_k15 = torch.cat(
                [dk15_m5, dk15_m4, dk15_m3, dk15_m2, dk15_m1, dist_tensor_k15, dk15_p1, dk15_p2, dk15_p3, dk15_p4,
                 dk15_p5], dim=0)
            offsets_k15 = hparams.v_shift - torch.argmax(dist_tensor_k15, dim=1)
            cur_num_correct_pred_k15_t1 = len(torch.where(abs(offsets_k15) <= 1)[0])
            cur_num_correct_pred_k15_t5 = len(torch.where(abs(offsets_k15) <= 5)[0])

            samplewise_acc_k15_t1.append(cur_num_correct_pred_k15_t1 / len(offsets_k15))
            samplewise_acc_k15_t5.append(cur_num_correct_pred_k15_t5 / len(offsets_k15))

            # K=20
            dist_tensor_k20 = (dist_tensor_k5[8:-7] + dist_tensor_k5[7:-8] + dist_tensor_k5[9:-6] + dist_tensor_k5[
                                                                                                    6:-9] +
                               dist_tensor_k5[10:-5] + dist_tensor_k5[5:-10] + dist_tensor_k5[11:-4] + dist_tensor_k5[
                                                                                                       4:-11] +
                               dist_tensor_k5[12:-3] + dist_tensor_k5[3:-12] + dist_tensor_k5[13:-2] + dist_tensor_k5[
                                                                                                       2:-13] +
                               dist_tensor_k5[14:-1] + dist_tensor_k5[1:-14] + dist_tensor_k5[15:] + dist_tensor_k5[
                                                                                                     :-15]) / 16
            dk20_m1 = torch.mean(dist_tensor_k5[:15], dim=0).unsqueeze(0)
            dk20_p1 = torch.mean(dist_tensor_k5[-15:], dim=0).unsqueeze(0)
            dk20_m2 = torch.mean(dist_tensor_k5[:14], dim=0).unsqueeze(0)
            dk20_p2 = torch.mean(dist_tensor_k5[-14:], dim=0).unsqueeze(0)
            dk20_m3 = torch.mean(dist_tensor_k5[:13], dim=0).unsqueeze(0)
            dk20_p3 = torch.mean(dist_tensor_k5[-13:], dim=0).unsqueeze(0)
            dk20_m4 = torch.mean(dist_tensor_k5[:12], dim=0).unsqueeze(0)
            dk20_p4 = torch.mean(dist_tensor_k5[-12:], dim=0).unsqueeze(0)
            dk20_m5 = torch.mean(dist_tensor_k5[:11], dim=0).unsqueeze(0)
            dk20_p5 = torch.mean(dist_tensor_k5[-11:], dim=0).unsqueeze(0)
            dk20_m6 = torch.mean(dist_tensor_k5[:10], dim=0).unsqueeze(0)
            dk20_p6 = torch.mean(dist_tensor_k5[-10:], dim=0).unsqueeze(0)
            dk20_m7 = torch.mean(dist_tensor_k5[:9], dim=0).unsqueeze(0)
            dk20_p7 = torch.mean(dist_tensor_k5[-9:], dim=0).unsqueeze(0)
            dk20_m8 = torch.mean(dist_tensor_k5[:8], dim=0).unsqueeze(0)

            dist_tensor_k20 = torch.cat([dk20_m8, dk20_m7, dk20_m6, dk20_m5, dk20_m4, dk20_m3, dk20_m2, dk20_m1,
                                         dist_tensor_k20,
                                         dk20_p1, dk20_p2, dk20_p3, dk20_p4, dk20_p5, dk20_p6, dk20_p7], dim=0)
            offsets_k20 = hparams.v_shift - torch.argmax(dist_tensor_k20, dim=1)
            cur_num_correct_pred_k20_t1 = len(torch.where(abs(offsets_k20) <= 1)[0])
            cur_num_correct_pred_k20_t5 = len(torch.where(abs(offsets_k20) <= 5)[0])

            samplewise_acc_k20_t1.append(cur_num_correct_pred_k20_t1 / len(offsets_k20))
            samplewise_acc_k20_t5.append(cur_num_correct_pred_k20_t5 / len(offsets_k20))

            # K=25
            dist_tensor_k25 = (dist_tensor_k5[10:-10] + dist_tensor_k5[9:-11] + dist_tensor_k5[11:-9] +
                               dist_tensor_k5[8:-12] + dist_tensor_k5[12:-8] + dist_tensor_k5[7:-13] +
                               dist_tensor_k5[13:-7] + dist_tensor_k5[6:-14] + dist_tensor_k5[14:-6] +
                               dist_tensor_k5[5:-15] + dist_tensor_k5[15:-5] + dist_tensor_k5[4:-16] +
                               dist_tensor_k5[16:-4] + dist_tensor_k5[3:-17] + dist_tensor_k5[17:-3] +
                               dist_tensor_k5[2:-18] + dist_tensor_k5[18:-2] + dist_tensor_k5[1:-19] +
                               dist_tensor_k5[19:-1] + dist_tensor_k5[:-20] + dist_tensor_k5[20:]) / 21
            dk25_m1 = torch.mean(dist_tensor_k5[:20], dim=0).unsqueeze(0)
            dk25_p1 = torch.mean(dist_tensor_k5[-20:], dim=0).unsqueeze(0)
            dk25_m2 = torch.mean(dist_tensor_k5[:19], dim=0).unsqueeze(0)
            dk25_p2 = torch.mean(dist_tensor_k5[-19:], dim=0).unsqueeze(0)
            dk25_m3 = torch.mean(dist_tensor_k5[:18], dim=0).unsqueeze(0)
            dk25_p3 = torch.mean(dist_tensor_k5[-18:], dim=0).unsqueeze(0)
            dk25_m4 = torch.mean(dist_tensor_k5[:17], dim=0).unsqueeze(0)
            dk25_p4 = torch.mean(dist_tensor_k5[-17:], dim=0).unsqueeze(0)
            dk25_m5 = torch.mean(dist_tensor_k5[:16], dim=0).unsqueeze(0)
            dk25_p5 = torch.mean(dist_tensor_k5[-16:], dim=0).unsqueeze(0)
            dk25_m6 = torch.mean(dist_tensor_k5[:15], dim=0).unsqueeze(0)
            dk25_p6 = torch.mean(dist_tensor_k5[-15:], dim=0).unsqueeze(0)
            dk25_m7 = torch.mean(dist_tensor_k5[:14], dim=0).unsqueeze(0)
            dk25_p7 = torch.mean(dist_tensor_k5[-14:], dim=0).unsqueeze(0)
            dk25_m8 = torch.mean(dist_tensor_k5[:13], dim=0).unsqueeze(0)
            dk25_p8 = torch.mean(dist_tensor_k5[-13:], dim=0).unsqueeze(0)
            dk25_m9 = torch.mean(dist_tensor_k5[:12], dim=0).unsqueeze(0)
            dk25_p9 = torch.mean(dist_tensor_k5[-12:], dim=0).unsqueeze(0)
            dk25_m10 = torch.mean(dist_tensor_k5[:11], dim=0).unsqueeze(0)
            dk25_p10 = torch.mean(dist_tensor_k5[-11:], dim=0).unsqueeze(0)

            dist_tensor_k25 = torch.cat(
                [dk25_m10, dk25_m9, dk25_m8, dk25_m7, dk25_m6, dk25_m5, dk25_m4, dk25_m3, dk25_m2, dk25_m1,
                 dist_tensor_k25,
                 dk25_p1, dk25_p2, dk25_p3, dk25_p4, dk25_p5, dk25_p6, dk25_p7, dk25_p8, dk25_p9, dk25_p10], dim=0)
            offsets_k25 = hparams.v_shift - torch.argmax(dist_tensor_k25, dim=1)
            cur_num_correct_pred_k25_t1 = len(torch.where(abs(offsets_k25) <= 1)[0])
            cur_num_correct_pred_k25_t5 = len(torch.where(abs(offsets_k25) <= 5)[0])

            samplewise_acc_k25_t1.append(cur_num_correct_pred_k25_t1 / len(offsets_k25))
            samplewise_acc_k25_t5.append(cur_num_correct_pred_k25_t5 / len(offsets_k25))

            prog_bar.set_description(
                '[Tolerance 1]:K5:{:.4f},K10:{:.4f},K15:{:.4f},K20:{:.4f},K25:{:.4f},[Tolerance 5]:K5:{:.4f},K10:{:.4f},K15:{:.4f},K20:{:.4f},K25:{:.4f}'
                    .format(np.mean(samplewise_acc_k5_t1),
                            np.mean(samplewise_acc_k10_t1),
                            np.mean(samplewise_acc_k15_t1),
                            np.mean(samplewise_acc_k20_t1),
                            np.mean(samplewise_acc_k25_t1),
                            np.mean(samplewise_acc_k5_t5),
                            np.mean(samplewise_acc_k10_t5),
                            np.mean(samplewise_acc_k15_t5),
                            np.mean(samplewise_acc_k20_t5),
                            np.mean(samplewise_acc_k25_t5)))

    return


def loadcheckpoint(model, checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    return model


if __name__ == "__main__":
    checkpoint_path = args.checkpoint_path
    # Dataset and Dataloader setup
    test_dataset = Dataset('test_unseen')
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        num_workers=0)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncTransformer().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    loadcheckpoint(model, checkpoint_path)
    with torch.no_grad():
        eval_model(test_data_loader, device, model)
