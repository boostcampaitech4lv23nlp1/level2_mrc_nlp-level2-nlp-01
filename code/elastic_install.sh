apt-get update && apt-get install -y gnupg2
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list
apt-get update && apt-get install elasticsearch
service elasticsearch start
cd /usr/share/elasticsearch
bin/elasticsearch-plugin install analysis-nori
service elasticsearch restart
pip install elasticsearch