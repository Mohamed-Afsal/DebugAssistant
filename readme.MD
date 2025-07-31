### Python Tips
rm -rf .venv/
vi requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 logs_debugger.py
export WML_API_KEY="xxxx-xxxx" (get from cloud )
export WML_PROJECT_ID=xxxxxxx-xxxx
pip install flask flask-cors
python3 app.py 

### SSH tunneling
  ssh -L 5000:<user@vm_domain_name>:5000 <user@accessible_vm_domain_name>

### docker tips
docker build -t debug-assistant .
docker run -d -p 5000:5000 \
  -e WML_API_KEY=xxx-xxx-xxx \
  -e WML_PROJECT_ID=xxxxx-xxx-xxx \
  -v ./grc.log:/app/grc.log \
  debug-assistant

### How to access in local machine browser
http://localhost:5000


