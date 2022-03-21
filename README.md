# Motion Gesture Detector

**Qinghui Gao**

**Yifan Ruan**

*This project accompanies the article [Motion gesture detection using Tensorflow on Android](http://blog.lemberg.co.uk/motion-gesture-detection-using-tensorflow-android)*

## Designed Gestures
- `MoveForward`
- `MoveRight`
- `MoveLeft`
- `MoveAround`

## Model
- Based on the article's model
- Like image recognition
- The input is treated as a 1 row * 128 columns * 3 channels image (128 sample points, 3 axis of accelerometer)
- Output is a vector of probabilities
- Input data are collected manually

### Diagram of our nerual networks
![Diagram of NN](https://lembergsolutions.com/sites/default/files/blog/imce/image_13.jpg)

## Algorithm (in brief)
- Multi threads for recognition and sensor
- Recognition creates a possibility output array
- The detector detects a new gesture coming
    - by comparing the possibilities with threshold
    - and judge if it is the same gesture in a short period
- Find the most possible gesture and return

## Improvements
- Consider 3 values of accelerometer: X, Y, Z, instead of only X and Y
    - Can recognize `MoveForward`
    - Improve stability
- Add motion delay
    - Won't detect the same gesture in short periods
- Improve logic
    - The former code outputs `MoveRight` if `MoveLeft` is under some threshold, which leads to many `MoveRight`s
    - We just find the maximum one, and leave the threshold judgement to other codes
- Code improvements
    - More versatile (use variable arguments and loops)
    - New features (lambda expression)

## Others
- REALLY HARD to configure environments
    - Conda packages are outdated, add cf201901 label to conda-forge
    - TOO MANY env conflicts related to glibc, tensorflow, bazel, ...
- Can't run on Android 11 due to conflicts between new features and library limits
- Use local `motiondetectionlib` instead of the original one

*It's not easy to be a system person. -- ryf*
