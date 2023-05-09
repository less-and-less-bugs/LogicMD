#lr=0.001
phase='train'
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-best-lr' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-best-lr' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5  &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='lr-0.0001-best-lr' --batch-size=32 --lr=0.0003 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5  &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-best-lr' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5  &

# find best threshold
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='find-logic-score' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=1  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5  --wd=0.001 &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-best-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.00005 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='lr-0.0001-best-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-best-wd' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.00001 &


# set instance dim rate
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-instancedim' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=300 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-instancedim' no--batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=100 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-rate-1.5' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.15 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-change-loss' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=300 &

# graph relu +
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-200-finetune' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='twotothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-200-finetune' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='onetothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0005 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-200-rate' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-clue-15' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=18  --log="weibo-lr-clue15"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-300-clue-15' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=18  --log="weibo-lr-clue15-instance300"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=300 &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='lr-0.0001-200-clue-10' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=18  --log="weibo-lr-clue10"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200 & sdf
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-200-clue-20' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=18  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=200 &

#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-200-rate' --batch-size=32 --lr=0.00007 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=200 &

#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.00005-graph-relu-300-lr' --batch-size=32 --lr=0.00005 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=300 &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-graph-relu-300-rate' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.15 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=300 &

# the effectiveness of hop
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-one' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-hop-one"   --hop='one' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-two' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-hop-two"   --hop='two' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-onetothree' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-hop-onetothree"   --hop='onetothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200 &
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-onetotwo' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-hop-onetotwo"   --hop='onetotwo' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-twotothree' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-hop-twotothree"   --hop='twotothree' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='lr-0.0001-200-hop-two' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=18  --log="weibo-lr-clue10"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200 &

# the number of clues
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='weibo-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-clue1"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='1#1#1#1#1'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='weibo-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-clue5"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='5#5#5#5#5'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='weibo-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-clue15"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='15#15#15#15#15'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='weibo'  --tag='weibo-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-clue25"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='25#25#25#25#25'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='weibo'  --tag='weibo-clue' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-clue20"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='20#20#20#20#20'  --topk=5 --wd=0.0001 --instancedim=200

# the rate
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.5' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-0.5"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.05 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='weibo'  --tag='lr-0.15' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-0.15"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.15 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.2' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-0.2"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.2 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='weibo'  --tag='lr-0.25' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="weibo-0.25"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.25 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200

#final model
CUDA_VISIBLE_DEVICES=2 python train_two_record_model.py --data='weibo'  --tag='final-model' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="final-model"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --type='fix' --sizeclues='10#10#10#10#10'  --topk=5 --wd=0.0001 --instancedim=200


