#lr=0.001
phase='train'






CUDA_VISIBLE_DEVICES=1 python drive_uamfd.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-debug"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0005 --type='fix' &
CUDA_VISIBLE_DEVICES=0 python drive_uamfd.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-debug"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.001 --type='fix' &
CUDA_VISIBLE_DEVICES=2 python drive_uamfd.py --data='weibo'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr-debug"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0005 --type='fix' &
CUDA_VISIBLE_DEVICES=3 python drive_uamfd.py --data='weibo'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr-debug"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.001 --type='fix' &


