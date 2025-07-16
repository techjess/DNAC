

import requests
import json
import time
from datetime import datetime, timedelta
import urllib3
import sqlite3

class DNACenterConnector:
    """
    My thought process: Create a class that handles all DNA Center interactions.
    Why a class? Because we need to maintain state (tokens, base URLs, etc.)
    """
    
    def __init__(self, base_url, username, password):
        """
        Why these parameters?
        - base_url: DNA Center changes URLs, make it configurable
        - username/password: Obviously needed, but we'll secure this later
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.username = username
        self.password = password
        self.token = None
        self.token_expiry = None
        
    def get_auth_token(self):
        """
        My approach: Always check if token is valid before making new requests
        Why? Tokens expire, and we don't want to hammer the auth endpoint
        """
        
        # Check if we have a valid token
        if self.token and self.token_expiry:
            if datetime.now() < self.token_expiry:
                print("Using existing valid token")
                return self.token
        
        print("Getting new authentication token...")
        
        # DNA Center auth endpoint
        auth_url = f"{self.base_url}/dna/system/api/v1/auth/token"
        
        # Headers for authentication
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            # Make the authentication request
            response = requests.post(
                auth_url,
                auth=(self.username, self.password),  # Basic auth
                headers=headers,
                verify=False  # Only for lab environments!
            )
            
            # Check if request was successful
            response.raise_for_status()  # Raises exception for bad status codes
            
            # Extract token from response
            token_data = response.json()
            self.token = token_data['Token']
            
            # DNA Center tokens typically last 1 hour, set expiry for 55 minutes
            self.token_expiry = datetime.now() + timedelta(minutes=55)
            
            print(f"✅ Authentication successful! Token expires at {self.token_expiry}")
            return self.token
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Authentication failed: {e}")
            raise
        except KeyError:
            print("❌ Invalid response format - check your credentials")
            raise
    
    def test_connection(self):
        """
        My philosophy: Always test connectivity before proceeding
        Why this endpoint? It's simple and tells us if our auth is working
        """
        
        # Get a valid token first
        token = self.get_auth_token()
        
        # Test endpoint - get network device count
        test_url = f"{self.base_url}/dna/intent/api/v1/network-device"
        
        headers = {
            'X-Auth-Token': token,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(test_url, headers=headers, verify=False)
            response.raise_for_status()
            
            devices = response.json()
            device_count = len(devices.get('response', []))
            
            print(f"✅ Connection test successful!")
            print(f"📊 Found {device_count} network devices")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
        
    # Add this method to our DNACenterConnector class

    def get_network_devices(self):
        """
        My approach: Start with basic device inventory
        Why this data first?
        1. Always available (even if devices are down)
        2. Foundational data for everything else  
        3. Easy to validate (you know your network)
        4. Small payload (won't timeout)
        """
        
        print("🔍 Collecting network device inventory...")
        
        # Get valid token
        token = self.get_auth_token()
        
        # DNA Center devices endpoint
        devices_url = f"{self.base_url}/dna/intent/api/v1/network-device"
        
        headers = {
            'X-Auth-Token': token,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(devices_url, headers=headers, verify=False)
            response.raise_for_status()
            
            devices_data = response.json()
            devices = devices_data.get('response', [])
            
            print(f"✅ Successfully collected {len(devices)} devices")
            
            # Let's process this data intelligently
            processed_devices = []
            
            for device in devices:
                # My approach: Extract only what we need, handle missing data gracefully
                processed_device = {
                    'device_id': device.get('id', 'unknown'),
                    'hostname': device.get('hostname', 'unknown'),
                    'management_ip': device.get('managementIpAddress', 'unknown'),
                    'platform': device.get('platformId', 'unknown'), 
                    'software_version': device.get('softwareVersion', 'unknown'),
                    'device_type': device.get('type', 'unknown'),
                    'role': device.get('role', 'unknown'),
                    'reachability': device.get('reachabilityStatus', 'unknown'),
                    'collected_at': datetime.now().isoformat()
                }
                
                processed_devices.append(processed_device)
            
            return processed_devices
            
        except Exception as e:
            print(f"❌ Failed to collect devices: {e}")
            raise

    def explore_api_response(self):
        """METHOD 4: Explore what data is available"""
        print("🔍 Exploring available device fields...")
        
        # Get valid token
        token = self.get_auth_token()
        
        # Get raw device data
        devices_url = f"{self.base_url}/dna/intent/api/v1/network-device"
        headers = {
            'X-Auth-Token': token,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(devices_url, headers=headers, verify=False)
            response.raise_for_status()
            
            devices_data = response.json()
            devices = devices_data.get('response', [])
            
            if devices:
                first_device = devices[0]
                available_keys = list(first_device.keys())
                
                print(f"📋 Found {len(available_keys)} fields in device data:")
                for key in sorted(available_keys):
                    value = first_device.get(key, 'N/A')
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:50] + "..."
                    print(f"  - {key}: {value}")
                
                return available_keys
            else:
                print("❌ No devices found")
                return []
                
        except Exception as e:
            print(f"❌ Failed to explore API: {e}")
            return []
    
    def get_device_health(self, device_id):
        """METHOD 5: Get real-time health data for specific device"""
        print(f"🏥 Getting health data for device: {device_id}")
        
        token = self.get_auth_token()
        
        # Health endpoint
        health_url = f"{self.base_url}/dna/intent/api/v1/device-health"
        
        headers = {
            'X-Auth-Token': token,
            'Content-Type': 'application/json'
        }
        
        # Add device filter
        params = {'deviceId': device_id}
        
        try:
            response = requests.get(health_url, headers=headers, params=params, verify=False)
            response.raise_for_status()
            
            health_data = response.json()
            return health_data.get('response', [])
            
        except Exception as e:
            print(f"❌ Failed to get device health: {e}")
            return None

    # Add this method to your existing DNACenterConnector class

    def get_device_health(self, device_id):
        """
        Gets real-time health data for a specific device
        We need device_id to tell DNA Center which device we want
        """
        
        print(f"🏥 Getting health data for device: {device_id}")
        
        # We use our existing get_auth_token method - no need to duplicate code
        token = self.get_auth_token()
        
        # The health endpoint - notice it's different from the device list endpoint
        health_url = f"{self.base_url}/dna/intent/api/v1/device-health"
        
        # We need to send the device ID as a parameter
        # Think of params like adding "?deviceId=xyz" to the end of the URL
        params = {'deviceId': device_id}
        
        headers = {
            'X-Auth-Token': token,
            'Content-Type': 'application/json'
        }
        
        try:
            # Notice we're adding params to our get request
            response = requests.get(health_url, headers=headers, params=params, verify=False)
            response.raise_for_status()  # This will throw an error if something went wrong
            
            health_data = response.json()
            return health_data.get('response', [])
            
        except Exception as e:
            print(f"❌ Failed to get device health: {e}")
            return None
        
    def setup_database(self):
        """
        Creates our fddatabase tables ifd they do not exist yet
        """
        print("Setting up database tables")

        # Connect to our database file(creates it if it does not exist)
        conn = sqlite3.connect('network_data.db')

        # A cursor is like a pen. WE use it to write to the database.
        cursor = conn.cursor()

        #Create our device_health table
        # This stores time stamped health data for each device. 
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS device_health (
            timestamp TEXT,
            device_id TEXT,
            hostname TEXT,
            cpu_utilization REAL,
            memory_utilization REAL,
            temperature REAL,
            reachability TEXT,
            collected_at TEXT
        )
    ''')
        
        # create our device inventory table
        # this stores basic device information
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS device_inventory (
            device_id TEXT PRIMARY KEY,
            hostname TEXT,
            platform TEXT,
            software_version TEXT,
            management_ip TEXT,
            device_type TEXT,
            last_updated TEXT
        )
    ''')
        
        # save our changes and close the connection
        conn.commit()
        conn.close()

        print("✅ Database tables ready!")
        
    def save_device_data(self, devices, health_data=None):
        """
        Saves our collected device data to the database
        devices = list of device info (from get_network_devices)
        health_data = optional health info for devices
        """
        
        print(f"💾 Saving data for {len(devices)} devices...")
        
        conn = sqlite3.connect('network_data.db')
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        # Save basic device inventory
        for device in devices:
            cursor.execute('''
                INSERT OR REPLACE INTO device_inventory 
                (device_id, hostname, platform, software_version, management_ip, device_type, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                device['device_id'],
                device['hostname'], 
                device['platform'],
                device['software_version'],
                device['management_ip'],
                device['device_type'],
                current_time
            ))
        
        # If we have health data, save that too
        if health_data:
            for device_id, health in health_data.items():
                # Extract health metrics (these might be None if not available)
                cpu = health.get('cpuUtilization') if health else None
                memory = health.get('memoryUtilization') if health else None
                temp = health.get('temperature') if health else None
                
                cursor.execute('''
                    INSERT INTO device_health 
                    (timestamp, device_id, cpu_utilization, memory_utilization, temperature, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    current_time,
                    device_id,
                    cpu,
                    memory, 
                    temp,
                    current_time
                ))
        
        conn.commit()
        conn.close()
        
        print("✅ Data saved to database!")

# Test our connector
if __name__ == "__main__":
    # Test with DNA Center sandbox (you'll replace with your details)
    dnac = DNACenterConnector(
        base_url="https://192.168.18.50",
        username="nsa01", 
        password="Crowbar-Parchment-Nail-Exes"
    )
    
    # Test the connection
    if dnac.test_connection():
        print("🎉 Connection successful!")
        
        # Set up our database first
        dnac.setup_database()
        
        # Get device data
        devices = dnac.get_network_devices()
        
        # Collect health data for all devices
        health_data = {}
        for device in devices[:3]:  # Just first 3 devices for testing
            device_id = device['device_id']
            health = dnac.get_device_health(device_id)
            if health:
                health_data[device_id] = health[0] if health else None
        
        # Save everything to database
        dnac.save_device_data(devices, health_data)
        
        print(f"🎯 Collected and stored data for {len(devices)} devices!")
