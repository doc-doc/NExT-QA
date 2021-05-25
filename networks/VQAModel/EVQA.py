import torch
import torch.nn as nn


class EVQA(nn.Module):
    def __init__(self, vid_encoder, qns_encoder, device):
        """

        :param vid_encoder:
        :param qns_encoder:
        :param ans_decoder:
        :param device:
        """
        super(EVQA, self).__init__()
        self.vid_encoder = vid_encoder
        self.qns_encoder = qns_encoder
        self.device = device
        self.FC = nn.Linear(qns_encoder.dim_hidden, 1)

    def forward(self, vid_feats, qas, qas_lengths):
        """

        :param vid_feats:
        :param qns:
        :param qns_lengths:
        :param mode:
        :return:
        """
        if self.qns_encoder.use_bert:
            cand_qas = qas.permute(1, 0, 2, 3)  # for BERT
        else:
            cand_qas = qas.permute(1, 0, 2)
        cand_len = qas_lengths.permute(1, 0)
        out = []
        for idx, qa in enumerate(cand_qas):
            encoder_out = self.vq_encoder(vid_feats, qa, cand_len[idx])
            out.append(encoder_out)

        out = torch.stack(out, 0).transpose(1, 0)

        _, predict_idx = torch.max(out, 1)


        return out, predict_idx

    def vq_encoder(self, vid_feats, qns, qns_lengths):

        vid_outputs, vid_hidden = self.vid_encoder(vid_feats)
        qns_outputs, qns_hidden = self.qns_encoder(qns, qns_lengths)


        qns_embed = qns_hidden[0].squeeze()
        vid_embed = vid_hidden[0].squeeze()

        fuse = qns_embed + vid_embed

        outputs = self.FC(fuse).squeeze()

        return outputs