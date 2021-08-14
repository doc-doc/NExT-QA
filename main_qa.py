from videoqa import *
import dataloader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_mc


def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 8
    else:
        batch_size = 4
        num_worker = 8
    spatial = False
    if spatial:
        #STVQA
        video_feature_path = '../data/feats/spatial/'
        video_feature_cache = '../data/feats/cache/'
    else:
        video_feature_cache = '../data/feats/cache/'
        video_feature_path = '../data/feats/'

    dataset = 'nextqa'
    sample_list_path = 'dataset/{}/'.format(dataset)
    vocab = pkload('dataset/{}/vocab.pkl'.format(dataset))

    glove_embed = 'dataset/{}/glove_embed.npy'.format(dataset)
    use_bert = True #Otherwise GloVe
    checkpoint_path = 'models'
    model_type = 'HGA' #(EVQA, CoMem, HME, HGA)
    model_prefix= 'bert-ft-h256'

    vis_step = 106
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 50


    data_loader = dataloader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab, use_bert, True, False)

    train_loader, val_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab, train_loader, val_loader, glove_embed, use_bert, checkpoint_path, model_type, model_prefix,
                  vis_step,lr_rate, batch_size, epoch_num)


    ep = 39
    acc = 49.64
    model_file = f'{model_type}-{model_prefix}-{ep}-{acc:.2f}.ckpt'

    if mode != 'train':
        result_file = f'results/{model_type}-{model_prefix}-{mode}.json'
        vqa.predict(model_file, result_file)
        eval_mc.main(result_file, mode)
    else:
        #Model for resume-training.
        model_file = f'{model_type}-{model_prefix}-0-00.00.ckpt'
        vqa.run(model_file, pre_trained=False)



if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--mode', dest='mode', type=str,
                        default='train', help='train or val')
    args = parser.parse_args()

    main(args)
