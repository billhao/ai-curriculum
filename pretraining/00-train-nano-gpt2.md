Follow Andrej Karpathy's video and trained nano GPT-2 (124M) base model. Spent 5 days to go through the video, replicate the code and finish the training.

Environment: 
  - mac (dev)
  - A100 (cuda and DDP testing)
  - 4 H800 (training) on AutoDL

Cost:
  - H800 @ 8.8 RMB/hour x 4 GPUs x 4 hrs = 140 RMB ($20 USD)

Data:
  - FineWebEdu 10B tokens

Training details:
  - Used same parameters as GPT-2 paper
  - Training takes 3.5-4 hrs
  - max steps = 19073
  - B = 64, T = 1024
  - Used ~40GB out of 80GB per GPU. There could be room to optimize and improve throughput. 
  
Throughput:
I can't get it to more than $1.1M tokens/s (~300k tokens/s per GPU). Increase B to 96 or 112 didn't help. OOM when B = 128.

Learning:
  - It was a lot fun to see the model crunching through so much data and compute and surpass GPT-2 loss and Hellaswag eval!
  - Real joy from the process
  - Learned so much on each piece in base model training
  - GPU cloud AutoDL is pretty easy to use
  - More confident now to dive deeper into models and components
