# Financial-Aid-Undergrad

## Dataset

Both raw and processed data is kept in the `datawarehouse` folder. Which however isn't committed due to data privacy. 

## Install

### AWS

If you are on an AWS EC2 instance, install a virtual python environment first.

```bash
sudo yum update -y
sudo yum install git -y
git clone https://github.com/UVA-MLSys/Financial-Aid.git

cd Financial-Aid
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SRGAsa2Hfy8dRxpwQUMbMQawTcvtDrqO' -O Merged.csv
```

sudo apt install python3.12
### Environment

Install the required libraries using the following command

```python
python3 -m pip install -r requirements.txt
```

## Run

```python
app.run_server(host= '0.0.0.0', port=8050)
gunicorn app:server --bind=0.0.0.0:8050
deactivate
```
