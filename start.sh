sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Overlay" exclusive_caps=1 max_buffers=2
python main.py $1