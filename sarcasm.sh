phase='train'

#CUDA_VISIBLE_DEVICES=0 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.0001' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="sarcasm-lr"   --hop='three' --finetune='false' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=0 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-seed"   --hop='three' --finetune='true' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=1 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-seed"   --hop='three' --finetune='true' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=2 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-15"   --hop='three' --finetune='true' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='15#15#15#15#15' &
CUDA_VISIBLE_DEVICES=2 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-final"   --hop='three' --finetune='true' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='10#10#10#10#10' &



#CUDA_VISIBLE_DEVICES=1 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-lr"   --hop='three' --finetune='true' --threshold=0.4 --rnn='true' --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=2 python train_sarcasm.py --data='sarcasm'  --tag='sarcasm-lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="sarcasm-lr"   --hop='three' --finetune='true' --threshold=0.4 --rnn='false' --rate=0.1 --sizeclues='10#10#10#10#10' &

#CUDA_VISIBLE_DEVICES=0 python train_two.py --data='twitter'  --tag='lr-0.0005' --batch-size=32 --lr=0.0005 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.00002' --batch-size=32 --lr=0.00002 --phase=$phase --epochs=20  --log="twitter-lr"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=1 python train_two.py --data='twitter'  --tag='lr-0.0001' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-lr-wd"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.1 --wd=0.001 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=2 python train_two.py --data='twitter'  --tag='rate-0.2' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.2"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.2 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.3' --batch-size=32 --lr=0.0001 --phase=$phase --epochs=20  --log="twitter-0.3"   --hop='three' --finetune='false' --threshold=0.4 --rate=0.3 --sizeclues='10#10#10#10#10' &
#CUDA_VISIBLE_DEVICES=3 python train_two.py --data='twitter'  --tag='rate-0.4' --batch-size=32 --lr=0.0001  --phase=$phase --epochs=20  --log="twitter-0.4"  --hop='three' --finetune='false'  --threshold=0.4 --rate=0.4 --sizeclues='10#10#10#10#10' &