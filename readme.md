install python 3.10

install pytorch based on your cuda. suggest 2.5+

install requirements.txt

download checkpoint and eval data from
`https://drive.google.com/drive/folders/1VH--WiX7jTraDrfwLaFFE93NEt38qqUm?usp=drive_link`


Put checkpoints under `checkpoints/`, like
`checkpoints/epoch00250_step083500/...[subfolders]`


Put data under `data/`, like `data/eval`


```bash
python Trans/main.py --config=configs/controlnet_eval.yaml
```

result will be in `output/`
