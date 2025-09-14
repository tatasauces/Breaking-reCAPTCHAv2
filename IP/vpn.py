import njord

client = njord.Client(user="", password="") #enter your username and password here
# Explicit
def connect():
    client.connect()

def disconnect():
    client.disconnect()
    print("vpn disconnected")