import argparse
from src.pipeline import ROOT,train
parser=argparse.ArgumentParser(description='Train YouTube popularity model')
parser.add_argument('--data',default=str(ROOT/'data'/'youtube_sample.csv'))
args=parser.parse_args()
_,metrics,df=train(args.data)
print(f'Clean records: {len(df)}')
for key,value in metrics.items(): print(f'{key}: {value:.3f}' if isinstance(value,float) else f'{key}: {value}')
print('Model and plots saved in models/ and outputs/')
