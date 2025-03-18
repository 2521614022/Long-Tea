import torch
import clip
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torch
from torch import nn
from PIL import Image
import os
import pandas as pd


def cls_acc(output, target, topk=2):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images) 
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha


def image_detect(img_path):
    
    print("\n-------- Classification of tea appearance comments. --------")
    
    pre_models_info = [
        ["finetune_resnet_straightness", 3], ["finetune_resnet_smoothness", 3],
        ["finetune_resnet_tenderness", 3], ["finetune_resnet_moisture", 2],
        ["finetune_resnet_fragmentation", 3], ["finetune_resnet_greenness", 2],
        ["finetune_resnet_flatness", 2], ["finetune_resnet_uniformity", 3],
        ["finetune_resnet_regression", 1]
    ]
    
    pre_models = []
    
    for i in range(len(pre_models_info)):
        pre_model = torchvision.models.resnet18(pretrained=False)
        pre_model.fc = nn.Linear(pre_model.fc.in_features, pre_models_info[i][1])
        pre_model.load_state_dict(torch.load("./caches/" + pre_models_info[i][0] + ".pth"))
        pre_model.eval()
        pre_models.append(pre_model)

    with torch.no_grad():
        # Create a converter to convert PIL images to Tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485,0.456,0.406],
                                             [0.229,0.224,0.225])
        ])
        test_image = transform(Image.open(os.path.join(img_path)))
        test_image = torch.unsqueeze(test_image, dim=0)
        
        predicted = []
        
        for i in range(len(pre_models)-1):
            output = pre_models[i](test_image)
            predicted.append(output.argmax(dim=1, keepdim=False))
        output = pre_models[-1](test_image)
        predicted.append(output)
        
    comment_labels = [
        ['Straightness', 'near straight', 'sharp', 'straight'],
        ['Smoothness', 'approach smooth', 'near smooth', 'smooth'],
        ['Tenderness', 'default', 'with buds', 'with some buds'],
        ['Moisture degree', 'bloom', 'near bloom'],
        ['Integral fragmentation', 'approach even', 'even', 'near even'],
        ['Greenness', 'green', 'near green'],
        ['Flatness', 'flat', 'near flat'],
        ['Color Uniformity', 'approach even', 'even', 'near even']
    ]
    
    for i in range(len(predicted)-1):
        print(f'{comment_labels[i][0]}: {comment_labels[i][predicted[i].item()+1]}')
        
    print("\n-------- Scoring based on image data. --------")
    print(f'shape_score: {predicted[-1].item():.2f}')
    
    return predicted
        
        
def table_detect(tab_path):
    
    print("\n-------- Scoring based on tabular data. --------")
    
    def get_net(i):
        if i == 0:
            net = nn.Sequential(nn.Linear(19, 256),nn.ReLU(),
                                nn.Linear(256, 128),nn.ReLU(),
                                nn.Linear(128, 64),nn.ReLU(),
                                nn.Linear(64, 1)) 
        elif i == 1:
            net = nn.Sequential(nn.Linear(313, 209),nn.ReLU(),
                                nn.Linear(209, 140),nn.ReLU(),
                                nn.Linear(140, 94),nn.ReLU(),
                                nn.Linear(94, 1)) 
        elif i == 2:
            net = nn.Sequential(nn.Linear(15, 256),nn.ReLU(),
                                nn.Linear(256, 128),nn.ReLU(),
                                nn.Linear(128, 64),nn.ReLU(),
                                nn.Linear(64, 1))
            
        elif i == 3:
            net = nn.Sequential(nn.Linear(15, 256),nn.ReLU(),
                                nn.Linear(256, 128),nn.ReLU(),
                                nn.Linear(128, 64),nn.ReLU(),
                                nn.Linear(64, 1))
        return net
    
    pre_models_info = [
        "MLP-D-4", "MLP-G-4", "MLP-TS-4", "MLP-YD-4"
    ]
    
    data = pd.read_csv(tab_path)

    dxzx_feature = torch.tensor(data[:19].values, dtype=torch.float32).reshape((-1,))
    gcms_feature = torch.tensor(data[19:332].values, dtype=torch.float32).reshape((-1,))
    ts_feature = torch.tensor(data[332:347].values, dtype=torch.float32).reshape((-1,))
    yd_feature = torch.tensor(data[347:].values, dtype=torch.float32).reshape((-1,))
    
    features = [dxzx_feature, gcms_feature, ts_feature, yd_feature]
    
    outputs = []
    
    for i in range(len(pre_models_info)):
        model = get_net(i)
        model.load_state_dict(torch.load("./caches/" + pre_models_info[i] + ".pt"))
        model.eval()
        with torch.no_grad():
            outputs.append(model(features[i]))

    table_scores = [
        ['dxzx_score', outputs[0]],
        ['gcms_score', outputs[1]],
        ['ts_score', outputs[2]],
        ['yd_score', outputs[3]]
    ]

    for i in range(len(table_scores)):
        print(f'{table_scores[i][0]}: {table_scores[i][1].item():.2f}')
        
    return table_scores


def tea_quality_classify(img_score, tab_score):
    
    print("\n-------- Classifying based on combinatorial data. --------")
    
    result = 0.25*img_score[-1].item() + 0.3*tab_score[0][1].item() + 0.25*tab_score[1][1].item() + 0.1*tab_score[2][1].item() + 0.1*tab_score[3][1].item()
    
    print(f'total_score: {result:.2f}')
    if result >= 90:
        print(f'The input is a kind of premium tea.')
    elif result < 88:
        print(f'The input is a kine of second grade tea.')
    else:
        print(f'The input is a kind of first grade tea.')