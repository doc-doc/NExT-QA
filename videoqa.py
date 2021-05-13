from networks import EncoderRNN, VQAModel, embed_loss
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import time


class VideoQA():
    def __init__(self, vocab, train_loader, val_loader, glove_embed, use_bert, checkpoint_path, model_type,
                 model_prefix, vis_step, lr_rate, batch_size, epoch_num):
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.glove_embed = glove_embed
        self.use_bert = use_bert
        self.model_dir = checkpoint_path
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.vis_step = vis_step
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):

        vid_dim = 2048 + 2048
        hidden_dim = 256
        word_dim = 300
        vocab_size = len(self.vocab)
        max_ans_len = 7
        max_vid_len = 16
        max_qa_len = 37

        if self.model_type == 'EVQA' or self.model_type == 'BlindQA':
            #ICCV15, AAAI17
            hidden_dim = 256
            vid_encoder = EncoderRNN.EncoderVid(vid_dim, hidden_dim, input_dropout_p=0.2, n_layers=1, rnn_dropout_p=0,
                                                bidirectional=False, rnn_cell='lstm')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1,
                                                input_dropout_p=0.2, rnn_dropout_p=0, bidirectional=False, rnn_cell='lstm')

            self.model = VQAModel.EVQA(vid_encoder, qns_encoder, self.device)

        elif self.model_type == 'STVQA':
            #CVPR17
            # vid_dim = 1024  # (64, 1024+2048, 7, 7)
            att_dim = 256
            vid_encoder = EncoderRNN.EncoderVidSTVQA(vid_dim, hidden_dim, input_dropout_p=0.2, rnn_dropout_p=0,
                                                     n_layers=1, rnn_cell='lstm')
            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert,
                                                input_dropout_p=0.2, rnn_dropout_p=0.5, n_layers=2, rnn_cell='lstm')

            self.model = VQAModel.STVQA(vid_encoder, qns_encoder, att_dim, self.device)


        elif self.model_type == 'CoMem':
            #CVPR18
            app_dim = 2048
            motion_dim = 2048
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.2,
                                                   bidirectional=False, rnn_cell='lstm')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=2,
                                                rnn_dropout_p=0.5, input_dropout_p=0.2, bidirectional=False, rnn_cell='lstm')

            self.model = VQAModel.CoMem(vid_encoder, qns_encoder, max_vid_len, max_qa_len, self.device)

        elif self.model_type == 'HME':
            #CVPR19
            app_dim = 2048
            motion_dim = 2048
            vid_encoder = EncoderRNN.EncoderVidCoMem(app_dim, motion_dim, hidden_dim, input_dropout_p=0.2,
                                                   bidirectional=False, rnn_cell='lstm')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=2,
                                                rnn_dropout_p=0.5, input_dropout_p=0.2, bidirectional=False, rnn_cell='lstm')


            self.model = VQAModel.HME(vid_encoder, qns_encoder, max_vid_len, max_qa_len, self.device)

        elif self.model_type == 'HGA':
            #AAAI20
            hidden_dim = 256
            vid_encoder = EncoderRNN.EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=0.3,
                                                     bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1,
                                                rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=False,
                                                rnn_cell='gru')

            self.model = VQAModel.HGA(vid_encoder, qns_encoder, self.device)


        params = [{'params':self.model.parameters()}]

        self.optimizer = torch.optim.Adam(params = params, lr=self.lr_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5, verbose=True)
        # Bugs to be fixed
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        # self.criterion = nn.CrossEntropyLoss() #yield similar results
        self.criterion = embed_loss.MultipleChoiceLoss().to(self.device)


    def save_model(self, epoch, acc):
        torch.save(self.model.state_dict(), osp.join(self.model_dir, '{}-{}-{}-{:.2f}.ckpt'
                                                     .format(self.model_type, self.model_prefix, epoch, acc)))

    def resume(self, model_file):
        """
        initialize model with pretrained weights
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        print(f'Warm-start (or test) with model {model_path}')
        model_dict = torch.load(model_path)
        new_model_dict = {}
        for k, v in self.model.state_dict().items():
            if k in model_dict:
                v = model_dict[k]
            else:
                pass
                # print(k)
            new_model_dict[k] = v
        self.model.load_state_dict(new_model_dict)


    def run(self, model_file, pre_trained=False):
        self.build_model()
        best_eval_score = 0.0
        if pre_trained:
            self.resume(model_file)
            best_eval_score = self.eval(0)
            print('Initial Acc {:.2f}'.format(best_eval_score))

        for epoch in range(1, self.epoch_num):
            train_loss, train_acc = self.train(epoch)
            eval_score = self.eval(epoch)
            print("==>Epoch:[{}/{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f}]".
                  format(epoch, self.epoch_num, train_loss, train_acc, eval_score))
            self.scheduler.step(eval_score)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                if epoch > 6 or pre_trained:
                    self.save_model(epoch, best_eval_score)


    def train(self, epoch):
        print('==>Epoch:[{}/{}][lr_rate: {}]'.format(epoch, self.epoch_num, self.optimizer.param_groups[0]['lr']))
        self.model.train()
        total_step = len(self.train_loader)
        epoch_loss = 0.0
        prediction_list = []
        answer_list = []
        for iter, inputs in enumerate(self.train_loader):
            videos, qas, qas_lengths, answers, qns_keys = inputs
            video_inputs = videos.to(self.device)
            qas_inputs = qas.to(self.device)
            ans_targets = answers.to(self.device)
            out, prediction = self.model(video_inputs, qas_inputs, qas_lengths)

            self.model.zero_grad()
            loss = self.criterion(out, ans_targets)
            loss.backward()
            self.optimizer.step()
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                print('\t[{}/{}]-{}-{:.4f}'.format(iter, total_step,cur_time, loss.item()))
            epoch_loss += loss.item()

            prediction_list.append(prediction)
            answer_list.append(answers)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers==ref_answers).numpy()

        return epoch_loss / total_step, acc_num*100.0 / len(ref_answers)


    def eval(self, epoch):
        print('==>Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        total_step = len(self.val_loader)
        acc_count = 0
        prediction_list = []
        answer_list = []
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                videos, qas, qas_lengths, answers, qns_keys = inputs
                video_inputs = videos.to(self.device)
                qas_inputs = qas.to(self.device)
                out, prediction = self.model(video_inputs, qas_inputs, qas_lengths)

                prediction_list.append(prediction)
                answer_list.append(answers)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()

        return acc_num*100.0 / len(ref_answers)


    def predict(self, model_file, result_file):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        self.build_model()
        if self.model_type in['HGA', 'STVQA']:
            self.resume(model_file)
        else:
            old_state_dict = torch.load(model_path)
            self.model.load_state_dict(old_state_dict)

        self.model.eval()
        results = {}
        with torch.no_grad():
            for iter, inputs in enumerate(self.val_loader):
                videos, qas, qas_lengths, answers, qns_keys = inputs
                video_inputs = videos.to(self.device)
                qas_inputs = qas.to(self.device)
                out, prediction = self.model(video_inputs, qas_inputs, qas_lengths)
                prediction = prediction.data.cpu().numpy()
                answers = answers.numpy()
                for qid, pred, ans in zip(qns_keys, prediction, answers):
                    results[qid] = {'prediction': int(pred), 'answer': int(ans)}


        print(len(results))
        save_file(results, result_file)









