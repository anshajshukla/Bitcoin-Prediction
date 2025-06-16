# Deployment Guide

## Local Development Setup

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure
```
bitcoin-timeseries-predictor/
├── data/
│   └── btc.csv
├── src/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
├── learning/
│   ├── README.md
│   ├── 01_Data_Preparation.md
│   ├── 02_Model_Implementation.md
│   ├── 03_Web_Interface.md
│   └── 04_Deployment_Guide.md
└── requirements.txt
```

### 3. Running Locally
```bash
# Navigate to project directory
cd bitcoin-timeseries-predictor

# Run the application
python run.py
```

## Production Deployment

### 1. Server Requirements
- Python 3.8+
- pip
- virtualenv
- nginx (recommended)
- gunicorn

### 2. Server Setup
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-venv nginx

# Create project directory
mkdir /var/www/bitcoin-prediction
cd /var/www/bitcoin-prediction

# Clone repository
git clone <repository-url> .

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Gunicorn Configuration
Create `gunicorn_config.py`:
```python
bind = "127.0.0.1:8000"
workers = 4
timeout = 120
```

### 4. Nginx Configuration
Create `/etc/nginx/sites-available/bitcoin-prediction`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /var/www/bitcoin-prediction/src/static;
    }
}
```

### 5. Systemd Service
Create `/etc/systemd/system/bitcoin-prediction.service`:
```ini
[Unit]
Description=Bitcoin Price Prediction
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/bitcoin-prediction
Environment="PATH=/var/www/bitcoin-prediction/venv/bin"
ExecStart=/var/www/bitcoin-prediction/venv/bin/gunicorn -c gunicorn_config.py src.app:app

[Install]
WantedBy=multi-user.target
```

### 6. Start Services
```bash
# Enable and start service
sudo systemctl enable bitcoin-prediction
sudo systemctl start bitcoin-prediction

# Enable and start nginx
sudo ln -s /etc/nginx/sites-available/bitcoin-prediction /etc/nginx/sites-enabled
sudo systemctl restart nginx
```

## Security Considerations

### 1. SSL/TLS
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 2. Firewall Configuration
```bash
# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443
```

### 3. Security Best Practices
1. Keep dependencies updated
2. Use environment variables for secrets
3. Implement rate limiting
4. Set up monitoring
5. Regular backups

## Monitoring and Maintenance

### 1. Logging
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
```

### 2. Monitoring
- Set up application monitoring
- Monitor server resources
- Track error rates
- Monitor prediction accuracy

### 3. Backup Strategy
1. Regular database backups
2. Configuration backups
3. Log backups
4. Disaster recovery plan

## Troubleshooting

### 1. Common Issues
1. Permission problems
2. Port conflicts
3. Memory issues
4. Database connection problems

### 2. Debugging
```bash
# Check service status
sudo systemctl status bitcoin-prediction

# View logs
sudo journalctl -u bitcoin-prediction

# Check nginx logs
sudo tail -f /var/log/nginx/error.log
```

## Scaling Considerations

### 1. Horizontal Scaling
- Load balancing
- Multiple workers
- Database replication
- Caching

### 2. Vertical Scaling
- Increase server resources
- Optimize database
- Improve caching
- Code optimization

## Maintenance Tasks

### 1. Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update system packages
sudo apt-get update
sudo apt-get upgrade
```

### 2. Monitoring
- Check server resources
- Monitor application logs
- Review error rates
- Track prediction accuracy

### 3. Backup
- Regular database backups
- Configuration backups
- Log rotation
- Disaster recovery testing 