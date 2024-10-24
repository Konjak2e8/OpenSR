lrs2="~/workspace/lrs2"
nshard=8
ffmpeg=ffmpeg
landmark_dir=${lrs2}/landmark

for step in $(seq 1 3); for rank in $(seq 0 $((nshard - 1)));python lrs2_prepare.py --lrs2 ${lrs2} --ffmpeg ${ffmpeg} --rank ${rank} --nshard ${nshard} --step ${step}
for rank in $(seq 0 $((nshard - 1)));python detect_landmark.py --root ${lrs2} --landmark ${lrs2}/landmark --manifest ${lrs2}/file.list \
 --cnn_detector ../../utils/mmod_human_face_detector.dat --face_detector ../../utils/shape_predictor_68_face_landmarks.dat --ffmpeg ${ffmpeg} \
 --rank ${rank} --nshard ${nshard}
for rank in $(seq 0 $((nshard - 1)));python align_mouth.py --video-direc ${lrs2} --landmark ${landmark_dir} --filename-path ${lrs2}/file.list \
 --save-direc ${lrs2}/video --mean-face ../../utils/20words_mean_face.npy --ffmpeg ${ffmpeg} \
 --rank ${rank} --nshard ${nshard}
for rank in $(seq 0 $((nshard - 1)));python count_frames.py --root ${lrs2} --manifest ${lrs2}/file.list --nshard ${nshard} --rank ${rank}
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/nframes.audio.${rank}; done > ${lrs2}/nframes.audio
for rank in $(seq 0 $((nshard - 1)));do cat ${lrs2}/nframes.video.${rank}; done > ${lrs2}/nframes.video
python lrs2_partioning.py --lrs2 ${lrs2}
for rank in $(seq 0 $((nshard - 1)));python lrs2_manifest.py --lrs2 ${lrs2} --manifest ${lrs2}/file.list \
 --valid-ids /path/to/valid --vocab-size ${vocab_size}
