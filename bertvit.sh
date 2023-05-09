#lr=0.001
phase='train'
#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='twitter'  --tag='-bert' --batch-size=32 --lr=0.001 --phase=$phase --epochs=20  --log="twitter-bert"   --finetune='false' --rnn='true' --modeltype='bert' --type='fix' &
#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='twitter'  --tag='-bert-vit' --batch-size=32 --lr=0.001 --phase=$phase --epochs=20  --log="twitter-bertvit"  --finetune='false' --rnn='true' --modeltype='vitbert' --type='fix' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001 --phase=$phase --epochs=20  --log="twitter-bert"  --finetune='false' --rnn='true' --modeltype='vit' --type='fix' &
#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-bert' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-bert"  --finetune='false' --rnn='true' --modeltype='bert' --type='fix' &
#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-bert-finetune' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-bert-finetune"  --finetune='true' --rnn='true' --modeltype='bert' --type='fix' &

#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-bert-vit' --batch-size=32 --lr=0.00002  --phase=$phase --epochs=20  --log="twitter-bert-vit" --finetune='false' --rnn='true' --type='vitbert' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0005  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' --wd=0.0005 &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' --wd=0.001 &


#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='twitter'  --tag='-bert-finetune' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-bert-finetune"   --finetune='false' --rnn='true' --type='bert' &
#CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='twitter'  --tag='-bert' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=20  --log="twitter-bert"  --finetune='false' --rnn='true' --type='bert' &
#CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='twitter'  --tag='-bert' --batch-size=32 --lr=0.001 --phase=$phase --epochs=20  --log="twitter-bert"  --finetune='false' --rnn='true' --type='bert' &
#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-bert' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-bert"  --finetune='false' --rnn='true' --type='bert' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-bert-vit' --batch-size=32 --lr=0.00002  --phase=$phase --epochs=20  --log="weibo-bert-vit" --finetune='false' --rnn='true' --type='vitbert' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0005  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' --wd=0.0005 &
#CUDA_VISIBLE_DEVICES=3 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001  --phase=$phase --epochs=20 --log="twitter-vit"  --finetune='false' --rnn='false' --type='vit' --wd=0.001 &

#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='weibo'  --tag='-bert' --batch-size=32 --lr=0.001 --phase=$phase --epochs=10  --log="weibo-bert"   --finetune='false' --rnn='true' --modeltype='bert' --type='fix' --wd=0.005 &
#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='weibo'  --tag='-bert' --batch-size=32 --lr=0.001 --phase=$phase --epochs=10  --log="weibo-bert"   --finetune='false' --rnn='true' --modeltype='bert' --type='fix' --wd=0.001 &
#CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='weibo'  --tag='-bert' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=10  --log="weibo-bert"   --finetune='false' --rnn='true' --modeltype='bert' --type='fix' --wd=0.0005 &

#CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='weibo'  --tag='-bert-vit' --batch-size=32 --lr=0.001 --phase=$phase --epochs=10  --log="weibo-bertvit"  --finetune='false' --rnn='true' --modeltype='vitbert' --type='fix' --wd=0.0005 &
#CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='weibo'  --tag='-bert-vit' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=10  --log="weibo-bertvit"  --finetune='false' --rnn='true' --modeltype='vitbert' --type='fix' --wd=0.001 &
#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='weibo'  --tag='-bert-vit' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=10  --log="weibo-bertvit"  --finetune='false' --rnn='true' --modeltype='vitbert' --type='fix' --wd=0.0005 &
#CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='weibo'  --tag='-bert-vit' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=10  --log="weibo-bertvit"  --finetune='false' --rnn='true' --modeltype='vitbert' --type='fix' --wd=0.001 &


CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.0005 --instancedim=64 &
CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.001 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.001 --instancedim=64 &
CUDA_VISIBLE_DEVICES=2 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.0005 --instancedim=64 &
CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.001 --instancedim=64 &
CUDA_VISIBLE_DEVICES=0 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.0005 --instancedim=64 &
CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.001 --instancedim=64 &
CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.0005 --instancedim=64 &
CUDA_VISIBLE_DEVICES=1 python drive_bertvit.py --data='twitter'  --tag='-vit' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=10  --log="twitter-bert"  --finetune='false' --rnn='false' --modeltype='bert' --type='fix' --wd=0.001 --instancedim=64 &
