# Koh 🎭 (aka "AI has many faces")
> Virtual camera that displays closest face match from a data set

## About

_You want to show your emotions on the internet but don't want to leak your identity? You want to see that you are just one of thousands of people? You want to express yourself with people from your favorite TV-episode?_ `Koh (aka "AI has many faces")` can help you with this! 

Koh analyses your face features from your webcam and shows the closest matching face from a given dataset. This creates an interesting effect, which you can show of as a virtual camera in your next video conference or live-stream. 

## Features

- [X] record webcam 👤🤳
- [X] track face point
- [X] train on face dataset 💪
- [X] get closest face match from dataset ⚖
- [X] use csv-files for better mixing datasets 🥗
- [ ] only match certain face features 🗜
- [ ] use match as virtual camera 📷

### Installation

> Look at [mediapipe/getting_startet](https://google.github.io/mediapipe/getting_started/python) 

> ❕ Make a virtual environment _(optional)_
```sh
python3 -m venv mp_env && source mp_env/bin/activate
```

Install dependencies

```sh
pip3 install -r requirements.txt
```
### Train

Download a face data set and save it in a folder in this directory. 

If you want to use a video make an image sequence out of it. Either use your video-editor of your choice (Blender) or use this ffmpeg command: 

```sh
ffmpeg -i "[video file]" dataset/[name]/image%04d.jpg
```

Input the relative path in the `train.py` script

```sh
./train.py [path/to/your/dataset]
```

## Usage

After you trained with your data set a file namend after the path should appear `[path_to_your_dataset].csv`. 

Input the relative path of that file into the `main.py`

```sh
./main.py [path_to_your_dataset].csv
```

If everthing went good two windows should appear. One shows your webcam with the wireframe of your face. The other window shows the matched face from your dataset. 

## Technologie

- [Mediapipe](https://mediapipe.dev/)
- OpenCV
- virtual camera

## Inspiration

- Koh - character from Avatar the last air bender
- [The Scramble Suit - A Scanner Darkly (2006)](https://youtu.be/2aS4xhTaIPc)
- ["Reflection" by Shane Cooper (ZKM)](https://zkm.de/de/werk/reflection)
- v-tuber, [FaceRig](https://facerig.com/)

## Authors

- **Gero Beckmann** - _Initial work_ - [Geronymos](https://github.com/Geronymos)

## License

This project is licensed under the GPT-3 License - see the `LICENSE` file for details
