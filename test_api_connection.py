#!/usr/bin/env python3
import requests
import socket
import subprocess
import os
import sys
import time

def get_network_info():
    """Gather and print network information"""
    print("=== Network Information ===")
    
    # Get hostname and local IP
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")
    
    try:
        local_ip = socket.gethostbyname(hostname)
        print(f"Local IP: {local_ip}")
    except Exception as e:
        print(f"Error getting local IP: {e}")
    
    # Get all IP addresses
    print("\nAll IP addresses:")
    try:
        import netifaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                print(f"  {interface}: {addrs[netifaces.AF_INET][0]['addr']}")
    except ImportError:
        print("netifaces module not available. Install with: pip install netifaces")
        # Try using ifconfig/ipconfig instead
        try:
            result = subprocess.check_output(['ifconfig' if os.name != 'nt' else 'ipconfig'], 
                                           stderr=subprocess.STDOUT,
                                           universal_newlines=True)
            print(result)
        except Exception as e:
            print(f"Error running ifconfig: {e}")

def check_port(host='0.0.0.0', port=5001):
    """Check if a port is open and not in use"""
    print(f"\n=== Port {port} Check ===")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        print(f"Port {port} is NOT in use (available)")
        result = False
    except socket.error:
        print(f"Port {port} is in use (unavailable)")
        result = True
    finally:
        s.close()
    return result

def test_api_connection(url="http://localhost:5001"):
    """Test connection to the API server"""
    print(f"\n=== Testing API Connection: {url} ===")
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"Connection successful! Response: {response.json()}")
            return True
        else:
            print(f"Connection failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Connection error. Server is not reachable.")
        return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

def test_multiple_urls(port=5001):
    """Test multiple URL combinations to find working connection"""
    # Get local IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    urls = [
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"http://{local_ip}:{port}",
        f"http://0.0.0.0:{port}"
    ]
    
    # Try to find Docker container IPs
    try:
        docker_ip_cmd = "hostname -I"
        docker_ips = subprocess.check_output(docker_ip_cmd, shell=True).decode().strip().split()
        for ip in docker_ips:
            if ip != local_ip and ip != "127.0.0.1":
                urls.append(f"http://{ip}:{port}")
    except:
        pass
    
    print("\n=== Testing multiple URL combinations ===")
    for url in urls:
        print(f"\nTrying: {url}")
        if test_api_connection(url):
            print(f"Success with URL: {url}")
            print(f"\nUse this URL in your client application: {url}")
            return url
    
    print("Failed to connect to API using any URL")
    return None

if __name__ == "__main__":
    print("API Connection Test Tool")
    get_network_info()
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 5001
    
    # First check if port is in use
    port_in_use = check_port(port=port)
    
    if port_in_use:
        # Try to connect to existing server
        test_multiple_urls(port)
    else:
        print(f"\nNo server detected on port {port}. Please start the server first.")
        
        # Offer to start server
        print("\nWould you like to start the server? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            print("Starting server...")
            server_cmd = f"cd {os.path.dirname(os.path.abspath(__file__))} && python apy.py --port {port} --no-preload-models"
            print(f"Running: {server_cmd}")
            
            # Start server in background
            subprocess.Popen(server_cmd, shell=True)
            
            # Wait for server to start
            print("Waiting for server to start...")
            time.sleep(5)
            
            # Test connection
            test_multiple_urls(port) 