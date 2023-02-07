# ABaG: Attention-Based Guide for grounded text-to-image generation
# Description
Attend-and-Exciteのlossを改造し、バウンディングボックスを用いた画像構図の操作を可能にします。<br>
BBox内のAttention map平均値を高め、BBox外のAttention map最大値を低めるように潜在空間を調整するだけのアイデアです。

# Usage
Example 1:
```
python run_abag.py --prompt "a mouse and a red car" --seeds [0] --token_indices [2,6] --bbox_txt_file ./bboxes/bbox1.txt --lr 0.6
```
|bbox1.txt means|generated image|
|--|--|
|![](https://github.com/birdManIkioiShota/ABaG/blob/main/images/bbox1.png)|![](https://github.com/birdManIkioiShota/ABaG/blob/main/images/a_mouse_and_a_car.png)|

Example 2:
```
python run_abag.py --prompt "a girl is eating pizza on desk" --seeds [0] --token_indices [2,5,7] --bbox_txt_file ./bboxes/bbox2.txt
```
|bbox2.txt means|generated image|
|--|--|
|![](https://github.com/birdManIkioiShota/ABaG/blob/main/images/bbox2.png)|![](https://github.com/birdManIkioiShota/ABaG/blob/main/images/a_girl_is_eating_pizza%20on_desk.png)|
