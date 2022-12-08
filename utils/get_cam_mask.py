from torch.nn import functional as F
import numpy as np
import cv2
def get_cam_mask(x, net, t1, model_name):
    # a batch(x,y),the classifier network resnet18
    # return mask where mask = 1 if cam of x <200 else mask = 0
    # 让网络关注更广泛的区域
    finalconv_name = 'layer4'
    net.eval()
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    handle = net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())

    if model_name == 'resnet_reparametrize':
        weight_softmax = np.squeeze(params[-8].data.cpu().numpy()) #有clf2
        mu = np.squeeze(params[-4].data.cpu().numpy()) 
    elif model_name == 'resnet':
        raise NotImplementedError
    else:
        raise ValueError()
    
    #weight_softmax = np.squeeze(params[-6].data.cpu().numpy()) #没有clf2
    #mu = np.squeeze(params[-2].data.cpu().numpy())    
    
    def returnCAM(feature_conv, weight_softmax, mu, idx):
        size_upsample = (224, 224)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for i in range(bz):
            cam = weight_softmax[idx[i][0]].dot(mu).dot(feature_conv[i,None,:,:].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / (np.max(cam)+1e-5)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return np.array(output_cam)
    logit,_ = net(x)
    logit = logit.cpu()
    h_x = F.softmax(logit, dim=-1).data.squeeze()
    probs, idx = h_x.sort(1, True)
    probs = probs.numpy()
    idx = idx.numpy()
    CAMs = returnCAM(features_blobs[0], weight_softmax,mu, idx) #输出概率最高的类的cam
    _,_,height, width = x.shape
    mask_unknow = np.ones(CAMs.shape)
    mask_unknow[np.where((CAMs > t1))] = 0
    
    handle.remove()
    return mask_unknow
