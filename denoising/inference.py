import os
import hydra
import logging

logger = logging.getLogger(__name__)



def run(args):
    import torch
    import torchaudio
    from torch.utils.data import DataLoader
    #from torch.utils.tensorboard import SummaryWriter
    print("CUDA??",torch.cuda.is_available())
    import soundfile as sf
    import datetime
    import numpy as np
    import scipy
    from tqdm import tqdm

    import utils.utils as utils 
    #import utils.lowpass_utils as lowpass_utils 
    #import  utils.dataset_loader as dataset_loader
    #import  utils.stft_loss as stft_loss
    #import models.discriminators as discriminators
    #import models.unet2d_generator as unet2d_generator
    #import models.audiounet as audiounet
    #import models.seanet as seanet
    import models.denoiser as denoiser

    #path_experiment=str(args.path_experiment)

    #if not os.path.exists(path_experiment):
    #    os.makedirs(path_experiment)
    
    #Loading data. The train dataset object is a generator. The validation dataset is loaded in memory.

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # ##NOT IMLEMENTED YET
    # if args.bwe.generator.variant=="audiounet": #change to audiounet
    #     #gener_model = kuleshov_unet.Unet1d(args.unet1d).to(device)
    #     gener_model = audiounet.Model(mono=True).to(device)
    # if args.bwe.generator.variant=="seanet": #change to seanet
    #     gener_model = seanet.Unet1d().to(device)
    # if args.bwe.generator.variant=="unet2d":
    #     gener_model = unet2d_generator.Unet2d(unet_args=args.unet_generator).to(device)

    dirname = os.path.dirname(__file__)
    checkpoint_filepath = os.path.join(dirname, str(args.checkpoint))

       
    #gener_model.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
    #print("something went wrong while loading the checkpoint")

    checkpoint_filepath_denoiser=os.path.join(dirname,str(args.checkpoint_denoiser))
    unet_model = denoiser.MultiStage_denoise(unet_args=args.denoiser)
    unet_model.load_state_dict(torch.load(checkpoint_filepath_denoiser, map_location=device))
    unet_model.to(device)



    def apply_denoiser_model(segment):
        segment_TF=utils.do_stft(segment,win_size=args.stft.win_size, hop_size=args.stft.hop_size, device=device)
        #segment_TF_ds=tf.data.Dataset.from_tensors(segment_TF)
        with torch.no_grad():
            pred = unet_model(segment_TF)
        if args.denoiser.num_stages>1:
            pred=pred[0]

        pred_time=utils.do_istft(pred, args.stft.win_size, args.stft.hop_size,device)
        #pred_time=pred_time[0]
        #pred_time=pred_time[0].detach().cpu().numpy()
        return pred_time

    try:
        audio=str(args.inference.audio)
        data, samplerate = sf.read(audio)
    except:
        print("reading relative path")
        audio=os.path.join(dirname,str(args.inference.audio))
        data, samplerate = sf.read(audio)
    #Stereo to mono
    if len(data.shape)>1:
        data=np.mean(data,axis=1)
    
    if samplerate!=args.fs: 
        print("Resampling")
   
        data=scipy.signal.resample(data, int((args.fs / samplerate )*len(data))+1)  
 
    
    segment_size=args.fs*args.seg_len_s_train  #5s segment

    length_data=len(data)
    overlapsize=int(args.stft.win_size*0.5) #samples (46 ms)
    window=np.hanning(2*overlapsize)
    window_right=window[overlapsize::]
    window_left=window[0:overlapsize]
    audio_finished=False
    pointer=0
    denoised_data=np.zeros(shape=(len(data),))
    denoised_lpf=np.zeros(shape=(len(data),))
    bwe_data=np.zeros(shape=(len(data),))
    numchunks=int(np.ceil(length_data/segment_size))

     
    for i in tqdm(range(numchunks)):
        if pointer+segment_size<length_data:
            segment=data[pointer:pointer+segment_size]
            #dostft
            segment = torch.from_numpy(segment)
            segment=segment.type(torch.FloatTensor)
            segment=segment.to(device)
            segment=torch.unsqueeze(segment,0)

            if args.inference.use_denoiser:
                denoised_time=apply_denoiser_model(segment)
                segment=denoised_time
                denoised_time=denoised_time[0].detach().cpu().numpy()
                #just concatenating with a little bit of OLA
                if pointer==0:
                    denoised_time=np.concatenate((denoised_time[0:int(segment_size-overlapsize)], np.multiply(denoised_time[int(segment_size-overlapsize):segment_size],window_right)), axis=0)
                else:
                    denoised_time=np.concatenate((np.multiply(denoised_time[0:int(overlapsize)], window_left), denoised_time[int(overlapsize):int(segment_size-overlapsize)], np.multiply(denoised_time[int(segment_size-overlapsize):int(segment_size)],window_right)), axis=0)
                denoised_data[pointer:pointer+segment_size]=denoised_data[pointer:pointer+segment_size]+denoised_time
            pointer=pointer+segment_size-overlapsize
        else: 
            segment=data[pointer::]

            lensegment=len(segment)
            segment=np.concatenate((segment, np.zeros(shape=(int(segment_size-len(segment)),))), axis=0)

            audio_finished=True
            #dostft
            segment = torch.from_numpy(segment)
            segment=segment.type(torch.FloatTensor)
            segment=segment.to(device)
            segment=torch.unsqueeze(segment,0)
            if args.inference.use_denoiser:
                denoised_time=apply_denoiser_model(segment)
                segment=denoised_time
                denoised_time=denoised_time[0].detach().cpu().numpy()
                if pointer!=0:
                    denoised_time=np.concatenate((np.multiply(denoised_time[0:int(overlapsize)], window_left), denoised_time[int(overlapsize):int(segment_size)]),axis=0)
                denoised_data[pointer::]=denoised_data[pointer::]+denoised_time[0:lensegment]

    
    basename=os.path.splitext(audio)[0]
    wav_noisy_name=basename+"_"+"_input"+".wav"
    sf.write(wav_noisy_name, data, args.fs)

    if args.inference.use_denoiser:
        wav_output_name=basename+"_"+"_denoised"+".wav"
        sf.write(wav_output_name, denoised_data, args.fs)


def _main(args):
    global __file__

    __file__ = hydra.utils.to_absolute_path(__file__)

    run(args)


@hydra.main(config_path="conf", config_name="conf")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()







