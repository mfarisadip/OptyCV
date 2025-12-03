#!/usr/bin/env python3
"""
GPS Module for OptyCV
Handles GPS data acquisition and coordinate conversion
"""
import serial
import time
import re
from typing import Optional, Tuple

class GPSManager:
    """Manage GPS data acquisition and coordinate conversion"""

    def __init__(self, port='/dev/ttyUSB3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.current_lat = None
        self.current_lng = None
        self.gps_active = False
        self.last_update_time = None

    def initialize_gps(self):
        """Initialize GPS connection and get first fix"""
        try:
            print(f"🛰️  Initializing GPS on {self.port}...")
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for GPS to stabilize

            # Clear any existing data
            self.ser.flushInput()

            # Get GPS fix
            success = self.get_gps_fix()
            if success:
                print(f"✅ GPS initialized successfully")
                print(f"📍 Current position: {self.current_lat:.6f}, {self.current_lng:.6f}")
                self.gps_active = True
                self.last_update_time = time.time()
                return True
            else:
                print("❌ Failed to get GPS fix")
                return False

        except Exception as e:
            print(f"❌ GPS initialization failed: {e}")
            return False

    def send_at_command(self, command: str, timeout: float = 1.0) -> bool:
        """Send AT command and wait for response"""
        try:
            self.ser.write((command + '\r\n').encode())
            time.sleep(timeout)

            response = ''
            if self.ser.inWaiting():
                response = self.ser.read(self.ser.inWaiting()).decode()

            return 'OK' in response
        except Exception as e:
            print(f"⚠️  GPS command error: {e}")
            return False

    def get_gps_fix(self, max_attempts: int = 10) -> bool:
        """Get GPS position fix"""
        print("🛰️  Getting GPS fix...")

        for attempt in range(max_attempts):
            try:
                # Send GPS commands
                self.send_at_command('AT+CGPS=0', 1)  # Stop GPS
                time.sleep(0.5)
                self.send_at_command('AT+CGPS=1', 1)  # Start GPS
                time.sleep(2)  # Wait for GPS fix

                # Get GPS info
                result = self.send_at_command('AT+CGPSINFO', 3)

                if result:
                    # Try to get NMEA data
                    self.ser.write('AT+CGPS=1,2\r\n'.encode())  # Enable NMEA output
                    time.sleep(2)

                    if self.ser.inWaiting():
                        nmea_data = self.ser.read(self.ser.inWaiting()).decode()

                        # Parse GPGGA sentence for lat/lng
                        gpgga_match = re.search(r'\$GPGGA,[^,]*,(\d+\.\d+),([NS]),(\d+\.\d+),([EW])', nmea_data)

                        if gpgga_match:
                            lat_str, lat_dir, lng_str, lng_dir = gpgga_match.groups()

                            # Convert to decimal degrees
                            lat = float(lat_str[:2]) + float(lat_str[2:]) / 60
                            lng = float(lng_str[:3]) + float(lng_str[3:]) / 60

                            # Apply direction
                            if lat_dir == 'S':
                                lat = -lat
                            if lng_dir == 'W':
                                lng = -lng

                            self.current_lat = lat
                            self.current_lng = lng

                            print(f"📍 GPS Fix obtained: {lat:.6f}°{lat_dir}, {lng:.6f}°{lng_dir}")
                            return True

                print(f"⏳ GPS attempt {attempt + 1}/{max_attempts}...")
                time.sleep(1)

            except Exception as e:
                print(f"⚠️  GPS attempt {attempt + 1} failed: {e}")
                time.sleep(1)

        print("❌ Failed to get GPS fix after all attempts")
        return False

    def update_position(self) -> bool:
        """Update current GPS position"""
        if not self.gps_active or not self.ser:
            return False

        try:
            # Quick position update
            result = self.send_at_command('AT+CGPSINFO', 1)

            if result and self.ser.inWaiting():
                data = self.ser.read(self.ser.inWaiting()).decode()

                # Parse for coordinates (simplified)
                coord_match = re.search(r'(\-?\d+\.\d+),(\-?\d+\.\d+)', data)
                if coord_match:
                    lat, lng = coord_match.groups()
                    self.current_lat = float(lat)
                    self.current_lng = float(lng)
                    self.last_update_time = time.time()
                    return True

        except Exception as e:
            print(f"⚠️  GPS update failed: {e}")

        return False

    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get current GPS coordinates"""
        if self.gps_active and self.current_lat and self.current_lng:
            # Update position if it's old (> 5 seconds)
            current_time = time.time()
            if self.last_update_time and (current_time - self.last_update_time) > 5:
                self.update_position()

            return (self.current_lat, self.current_lng)
        return None

    def get_gps_info_text(self) -> str:
        """Get formatted GPS info for display"""
        coords = self.get_coordinates()
        if coords:
            lat, lng = coords
            return f"GPS: {lat:.6f}, {lng:.6f}"
        else:
            return "GPS: No Fix"

    def close(self):
        """Close GPS connection"""
        if self.ser:
            self.ser.close()
            print("🛰️  GPS connection closed")