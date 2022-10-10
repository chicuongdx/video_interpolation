import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def load_model(path_model, device_name):
    model_path = path_model #'models/RRDB_ESRGAN_x4.pth'
    device = torch.device(device_name)  # cuda or cpu

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    return model, device

def process_frame(img, model, device):

    #frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    return output

def run_video(url=0):
    dsize = (3840, 2880)
    out = cv2.VideoWriter('results/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, dsize)
    cap = cv2.VideoCapture(url)
    model, device = load_model('models/RRDB_ESRGAN_x4.pth', 'cuda')
    idx = 0
    while True:
        try:
            ret, frame = cap.read()

            if ret==False:
                break

            frame_LR = process_frame(frame, model, device)
            #frame_LR = frame
            frame_LR = cv2.resize(frame_LR, dsize)
            out.write(frame_LR)
            print('frame in index: ' + str(idx) + ' ' + str(frame_LR.shape))
            idx+=1
        except NameError:
            print(NameError)
            break
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

run_video('FapsuJapan.mp4')