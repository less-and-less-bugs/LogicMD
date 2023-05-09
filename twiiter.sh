#lr=0.001
phase='train'

#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='twitter'  --tag='lr-0.0001' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='twitter'  --tag='lr-0.0001' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.00002-lr' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-wd"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0005 --type='fix' &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-wd"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0005 --type='fix' &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-wd"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.001 --type='fix' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='lr-0.00002-bert' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='true' --threshold=0.4 --rate=0.1 --type='fix' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='true' --threshold=0.4 --rate=0.1 --type='fix' &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.2' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.2"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.2 --type='fix' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.3' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.3"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.3 --type='fix' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.4' --batch-size=32 --lr=0.0001  --phase=$phase --epochs=20  --log="twitter-0.4"  --hop='three' --finetune='false'  --threshold=0.4 --rate=0.4 --type='fix' &

#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='rate-0.5' --batch-size=32 --lr=0.0001  --phase=$phase --epochs=20 --log="twitter-0.5"  --hop='three' --finetune='false' --threshold=0.4 --rate=0.5 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.6' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.6"   --hop='three' --finetune='false' --threshold=0.4  --rate=0.6 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.7' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.7"   --hop='three' --finetune='false' --threshold=0.4  --rate=0.7 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.8' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.8"   --hop='three' --finetune='false' --threshold=0.4  --rate=0.8 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.9' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.9"   --hop='three' --finetune='false' --threshold=0.4  --rate=0.9 &

#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='rate-0.5' --batch-size=32 --lr=0.0001  --phase=$phase --epochs=20 --log="twitter-0.5"  --hop='one' --finetune='false' --threshold=0.4 --rate=0.1 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.6' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.6"   --hop='two' --finetune='false' --threshold=0.4  --rate=0.1 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.7' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.7"   --hop='three' --finetune='false' --threshold=0.4  --rate=0.1 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.8' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.8"   --hop='onetwo' --finetune='false' --threshold=0.4  --rate=0.1 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.9' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.9"   --hop='onethree' --finetune='false' --threshold=0.4  --rate=0.1 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.9' --batch-size=32 --lr=0.0001   --phase=$phase --epochs=20 --log="twitter-0.9"   --hop='twothree' --finetune='false' --threshold=0.4  --rate=0.1 &


#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-wd"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0005 --type='fix' &


#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-final"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0001 --type='fix' --instancedim=200 &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='twitter'  --tag='lr-0.0001-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-final"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.0001 --type='fix' --instancedim=200 &


# the hop
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='twitter'  --tag='lr-0.0001-200-hop-one' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-hop-one"   --hop='one' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001-200-hop-two' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-hop-two"   --hop='two' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='lr-0.0001-200-hop-onetothree' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-hop-onetothree"   --hop='onetothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001-200-hop-onetotwo' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-hop-onetotwo"   --hop='onetotwo' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='lr-0.0001-200-hop-twotothree' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-hop-twotothree"   --hop='twotothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200

# the clue
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='twitter-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="clue1"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='1#1#1#1#1'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='twitter-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="clue5"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='5#5#5#5#5'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='twitter-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="clue15"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='twitter-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="clue20"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0005 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='twitter-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="clue25"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='25#25#25#25#25'  --topk=5 --wd=0.0005 --instancedim=200

# the rate
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.5' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.5"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.05 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.15' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.15"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.15 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.2' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.2"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.2 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.25' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.25"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.25 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200

# final twitter
CUDA_VISIBLE_DEVICES=1 python train_two_record_model.py --data='twitter'  --tag='final-model' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-final-model"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200
