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
import os, random, cv2, argparse
from hparams import hparams, get_image_list
from natsort import natsorted
import soundfile as sf
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(description='Code to test VocaLiST: the lip-sync detector on LRS2')
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset",
                    default="/mnt/DATA/dataset/LRS2/preprocessed")
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',
                    default="/mnt/DATA/dataset/acapsol/experiments/vocalist_weights/vocalist_5f_lrs2.pth",
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
        self.all_videos = get_image_list(args.data_root, split)

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
            wavpath = join(vidname, "audio.wav")
            img_names = natsorted(list(glob(join(vidname, '*.jpg'))), key=lambda y: y.lower())
            wav = self.get_wav(wavpath)
            min_length = min(len(img_names), math.floor(len(wav) / 640))
            lastframe = min_length - v_context

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
            if vid==None or mels==None:
                continue
            return vid, mels, lastframe


def calc_pdist(model, feat1, feat2, vshift=15):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2.permute(1,2,3,0).contiguous(), (vshift,vshift)).permute(3,0,1,2).contiguous()

    dists = []
    num_rows_dist = len(feat1)
    for i in range(0, num_rows_dist):
        raw_sync_scores = model(feat1[i].unsqueeze(0).repeat(win_size, 1, 1, 1).to(device), feat2p[i:i + win_size, :].to(device))
        dist_measures = raw_sync_scores.clone().cpu()
        if i in range(vshift):
            dist_measures[0:vshift-i] = torch.tensor(-1000, dtype=torch.float).to(device)
        elif i in range(num_rows_dist-vshift,num_rows_dist):
            dist_measures[vshift+num_rows_dist-i:] = torch.tensor(-1000, dtype=torch.float).to(device)

        dists.append(dist_measures)

    return dists


def eval_model(test_data_loader, device, model):
    prog_bar = tqdm(enumerate(test_data_loader))
    samplewise_acc_k5, samplewise_acc_k7, samplewise_acc_k9, samplewise_acc_k11, samplewise_acc_k13, samplewise_acc_k15 = [],[],[],[],[],[]
    for step, (vid, aud, lastframe) in prog_bar:
        model.eval()
        with torch.no_grad():
            vid = vid.view(BATCH_SIZE,(lastframe+v_context),3,48,96)
            batch_size = 20
            lastframe = lastframe.item()
            lim_in = []
            lcc_in = []
            for i in range(0, lastframe, batch_size):
                im_batch = [vid[:, vframe:vframe + v_context, :, :, :].view(BATCH_SIZE, -1, 48, 96) for vframe in
                            range(i, min(lastframe, i + batch_size))]
                im_in = torch.cat(im_batch, 0)
                lim_in.append(im_in)

                cc_batch = [aud[:, :, :, int(80.*(vframe/float(hparams.fps))):int(80.*(vframe/float(hparams.fps)))+mel_step_size] for vframe in
                            range(i, min(lastframe, i + batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                lcc_in.append(cc_in)

            lim_in = torch.cat(lim_in, 0)
            lcc_in = torch.cat(lcc_in, 0)
            dists = calc_pdist(model, lim_in, lcc_in, vshift=hparams.v_shift)

            # K=5
            dist_tensor_k5 = torch.stack(dists)
            offsets_k5 = hparams.v_shift - torch.argmax(dist_tensor_k5, dim=1)
            cur_num_correct_pred_k5 = len(torch.where(offsets_k5 == -1)[0]) + len(torch.where(offsets_k5 == 0)[0]) + len(
                torch.where(offsets_k5 == 1)[0])
            samplewise_acc_k5.append(cur_num_correct_pred_k5 / len(offsets_k5))

            # K=7
            dist_tensor_k7 = (dist_tensor_k5[1:-1] + dist_tensor_k5[2:] + dist_tensor_k5[:-2]) / 3  # inappropriate to average over 0,0,20 for example
            dk7_m1 = torch.mean(dist_tensor_k5[:2], dim=0).unsqueeze(0)
            dk7_p1 = torch.mean(dist_tensor_k5[-2:], dim=0).unsqueeze(0)
            dist_tensor_k7 = torch.cat([dk7_m1, dist_tensor_k7, dk7_p1], dim=0)
            offsets_k7 = hparams.v_shift - torch.argmax(dist_tensor_k7, dim=1)
            cur_num_correct_pred_k7 = len(torch.where(offsets_k7 == -1)[0]) + len(torch.where(offsets_k7 == 0)[0]) + len(torch.where(offsets_k7 == 1)[0])
            samplewise_acc_k7.append(cur_num_correct_pred_k7 / len(offsets_k7))

            # K=9
            dist_tensor_k9 = (dist_tensor_k5[2:-2] + dist_tensor_k5[1:-3] + dist_tensor_k5[3:-1] + dist_tensor_k5[:-4] + dist_tensor_k5[4:]) / 5
            dk9_m1 = torch.mean(dist_tensor_k5[:4], dim=0).unsqueeze(0)
            dk9_p1 = torch.mean(dist_tensor_k5[-4:], dim=0).unsqueeze(0)
            dk9_m2 = torch.mean(dist_tensor_k5[:3], dim=0).unsqueeze(0)
            dk9_p2 = torch.mean(dist_tensor_k5[-3:], dim=0).unsqueeze(0)
            dist_tensor_k9 = torch.cat([dk9_m2, dk9_m1, dist_tensor_k9, dk9_p1, dk9_p2], dim=0)
            offsets_k9 = hparams.v_shift - torch.argmax(dist_tensor_k9, dim=1)
            cur_num_correct_pred_k9 = len(torch.where(offsets_k9 == -1)[0]) + len(
                torch.where(offsets_k9 == 0)[0]) + len(torch.where(offsets_k9 == 1)[0])
            samplewise_acc_k9.append(cur_num_correct_pred_k9 / len(offsets_k9))

            # K=11
            dist_tensor_k11 = (dist_tensor_k5[3:-3] + dist_tensor_k5[2:-4] + dist_tensor_k5[4:-2] +
                               dist_tensor_k5[1:-5] + dist_tensor_k5[5:-1] + dist_tensor_k5[:-6] + dist_tensor_k5[6:]) / 7
            dk11_m1 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk11_p1 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)
            dk11_m2 = torch.mean(dist_tensor_k5[:5], dim=0).unsqueeze(0)
            dk11_p2 = torch.mean(dist_tensor_k5[-5:], dim=0).unsqueeze(0)
            dk11_m3 = torch.mean(dist_tensor_k5[:4], dim=0).unsqueeze(0)
            dk11_p3 = torch.mean(dist_tensor_k5[-4:], dim=0).unsqueeze(0)
            dist_tensor_k11 = torch.cat([dk11_m3, dk11_m2, dk11_m1, dist_tensor_k11, dk11_p1, dk11_p2, dk11_p3], dim=0)
            offsets_k11 = hparams.v_shift - torch.argmax(dist_tensor_k11, dim=1)
            cur_num_correct_pred_k11 = len(torch.where(offsets_k11 == -1)[0]) + len(
                torch.where(offsets_k11 == 0)[0]) + len(torch.where(offsets_k11 == 1)[0])
            samplewise_acc_k11.append(cur_num_correct_pred_k11 / len(offsets_k11))

            # K=13
            dist_tensor_k13 = (dist_tensor_k5[4:-4] + dist_tensor_k5[3:-5] + dist_tensor_k5[5:-3] +
                               dist_tensor_k5[2:-6] + dist_tensor_k5[6:-2] + dist_tensor_k5[1:-7] +
                               dist_tensor_k5[7:-1] + dist_tensor_k5[:-8] + dist_tensor_k5[8:]) / 9
            dk13_m1 = torch.mean(dist_tensor_k5[:8], dim=0).unsqueeze(0)
            dk13_p1 = torch.mean(dist_tensor_k5[-8:], dim=0).unsqueeze(0)
            dk13_m2 = torch.mean(dist_tensor_k5[:7], dim=0).unsqueeze(0)
            dk13_p2 = torch.mean(dist_tensor_k5[-7:], dim=0).unsqueeze(0)
            dk13_m3 = torch.mean(dist_tensor_k5[:6], dim=0).unsqueeze(0)
            dk13_p3 = torch.mean(dist_tensor_k5[-6:], dim=0).unsqueeze(0)
            dk13_m4 = torch.mean(dist_tensor_k5[:5], dim=0).unsqueeze(0)
            dk13_p4 = torch.mean(dist_tensor_k5[-5:], dim=0).unsqueeze(0)

            dist_tensor_k13 = torch.cat([dk13_m4, dk13_m3, dk13_m2, dk13_m1, dist_tensor_k13, dk13_p1, dk13_p2, dk13_p3, dk13_p4], dim=0)
            offsets_k13 = hparams.v_shift - torch.argmax(dist_tensor_k13, dim=1)
            cur_num_correct_pred_k13 = len(torch.where(offsets_k13 == -1)[0]) + len(
                torch.where(offsets_k13 == 0)[0]) + len(torch.where(offsets_k13 == 1)[0])
            samplewise_acc_k13.append(cur_num_correct_pred_k13 / len(offsets_k13))

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

            dist_tensor_k15 = torch.cat([dk15_m5, dk15_m4, dk15_m3, dk15_m2, dk15_m1, dist_tensor_k15, dk15_p1, dk15_p2, dk15_p3, dk15_p4, dk15_p5], dim=0)
            offsets_k15 = hparams.v_shift - torch.argmax(dist_tensor_k15, dim=1)
            cur_num_correct_pred_k15 = len(torch.where(offsets_k15 == -1)[0]) + len(
                torch.where(offsets_k15 == 0)[0]) + len(torch.where(offsets_k15 == 1)[0])
            samplewise_acc_k15.append(cur_num_correct_pred_k15 / len(offsets_k15))

            prog_bar.set_description('K5:{:.4f},K7:{:.4f},K9:{:.4f},K11:{:.4f},K13:{:.4f},K15:{:.4f}'
                                     .format(np.mean(samplewise_acc_k5),
                                             np.mean(samplewise_acc_k7),
                                             np.mean(samplewise_acc_k9),
                                             np.mean(samplewise_acc_k11),
                                             np.mean(samplewise_acc_k13),
                                             np.mean(samplewise_acc_k15)))

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
    test_dataset = Dataset('test')
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
