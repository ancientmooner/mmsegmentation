```
# Download converted model
# model is converted with tools/convert_dnl.py
wget https://openmmlab.oss-accelerate.aliyuncs.com/mmsegmentation/v0.5/dnl_debug/dnl_r101-d8_769x769_benchmark_cityscapes.pth

# test with
./tools/dist_test.sh configs/dnlnet/dnl_r101-d8_769x769_benchmark_cityscapes.py dnl_r101-d8_769x769_benchmark_cityscapes.pth 4 --eval mIoU --tmpdir tmpdir

mIoU: 78.49
```