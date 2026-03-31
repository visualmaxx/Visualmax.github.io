import sys
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

input_path = sys.argv[1]
output_path = sys.argv[2]
mode = sys.argv[3]

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64)
upsampler = RealESRGANer(scale=4, model_path='weights/RealESRGAN_x4plus.pth', model=model)

img = cv2.imread(input_path, cv2.IMREAD_COLOR)
output, _ = upsampler.enhance(img, outscale=4)
cv2.imwrite(output_path, output)
