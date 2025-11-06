from pydub import AudioSegment
import numpy as np
import glob
import os
import csv
segment_length = 3000 #Sliced time in milliseconds
#path_to_files="/media/nacho/RespaldoIIE/FING/Doctorado/Datasets/TapeNoisedB/uher_Report_4000L_3.75ips" 
path_to_files="mag_tape_db/tape_noise44100" 
out_folder='mag_tape_db_AudioSlice/'
validationSplit=0.9
# iterate over files in
# that directory
os.mkdir(out_folder)
with open(os.path.join(out_folder,'info.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    header=["recording","number_of_segments","total_length","max_length","largest_segment","split","segments","year"]
    writer.writerow(header)
    for filename in glob.iglob(f'{path_to_files}/*.wav'):
        #print(filename)
        newAudio = AudioSegment.from_wav(filename)
        numFiles=int(len(newAudio)/segment_length)
        for i in range(numFiles):
            out_path=os.path.join(out_folder ,os.path.basename(filename))
        #    os.mkdir(f'{os.path.basename(filename)}_{i}')
            os.mkdir(f'{out_path}_{i}')
            s = np.random.uniform()
            if 1>s>validationSplit:
                data_CSV=[f'{os.path.basename(filename)}_{i}',1,segment_length/1000,segment_length/1000,"segment-01.wav","validation","['segment-01.wav']","1964"]
            else:
                data_CSV=[f'{os.path.basename(filename)}_{i}',1,segment_length/1000,segment_length/1000,"segment-01.wav","train","['segment-01.wav']","1964"]
            #if 0.95>s>0.9:
            #data_CSV=[f'{os.path.basename(filename)}_{i}',1,segment_length/1000,segment_length/1000,"segment-01.wav","test","['segment-01.wav']","1964"]
            #if 0.9>s>0:
            #    data_CSV=[f'{os.path.basename(filename)}_{i}',1,segment_length/1000,segment_length/1000,"segment-01.wav","train","['segment-01.wav']","1964"]
            
            writer.writerow(data_CSV)
            segment = newAudio[i*segment_length:(i+1)*segment_length]
            #segment = segment.from_mono_audiosegments(segment,segment)
         #   segment.export(f'{os.path.basename(filename)}_{i}/segment-01.wav', format="wav")
            segment.export(f'{out_path}_{i}/segment-01.wav', format="wav")


