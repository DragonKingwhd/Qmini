"""Standalone test script for GY-91 (MPU9250 + BMP280) over I2C on Raspberry Pi.

Wiring:
    GY-91 VCC  -> Pi 3V3  (pin 1)
    GY-91 GND  -> Pi GND  (pin 6)
    GY-91 SDA  -> Pi SDA1 / GPIO2 (pin 3)
    GY-91 SCL  -> Pi SCL1 / GPIO3 (pin 5)

Prerequisites:
    sudo raspi-config -> Interface Options -> I2C -> Enable
    sudo apt install -y i2c-tools python3-smbus
    pip install smbus2

Run:
    python3 -m sim2real.deploy.tests.test_imu_gy91
"""

import time
import struct

try:
    from smbus2 import SMBus  # pip install smbus2 (preferred)
except ImportError:
    from smbus import SMBus   # apt install python3-smbus (fallback)

I2C_BUS = 1

# ---------- MPU9250 ----------
MPU9250_ADDR = 0x68
PWR_MGMT_1 = 0x6B
WHO_AM_I = 0x75
ACCEL_XOUT_H = 0x3B
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
SMPLRT_DIV = 0x19
CONFIG = 0x1A

ACCEL_SCALE = 16384.0  # ±2g, LSB/g
GYRO_SCALE = 131.0     # ±250 dps, LSB/(deg/s)

# ---------- BMP280 ----------
BMP280_ADDR = 0x76  # try 0x77 if 0x76 not found
BMP280_ID_REG = 0xD0
BMP280_RESET = 0xE0
BMP280_CTRL_MEAS = 0xF4
BMP280_CONFIG = 0xF5
BMP280_PRESS_MSB = 0xF7
BMP280_CALIB = 0x88


def init_mpu9250(bus: SMBus) -> None:
    who = bus.read_byte_data(MPU9250_ADDR, WHO_AM_I)
    print(f"[MPU9250] WHO_AM_I = 0x{who:02X} (expected 0x71 for MPU9250, 0x68 for MPU6050)")
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x00)  # wake
    time.sleep(0.1)
    bus.write_byte_data(MPU9250_ADDR, PWR_MGMT_1, 0x01)  # PLL with X gyro
    bus.write_byte_data(MPU9250_ADDR, SMPLRT_DIV, 0x00)
    bus.write_byte_data(MPU9250_ADDR, CONFIG, 0x03)       # DLPF 41Hz
    bus.write_byte_data(MPU9250_ADDR, GYRO_CONFIG, 0x00)  # ±250 dps
    bus.write_byte_data(MPU9250_ADDR, ACCEL_CONFIG, 0x00) # ±2g
    time.sleep(0.05)


def _to_int16(hi: int, lo: int) -> int:
    val = (hi << 8) | lo
    return val - 65536 if val & 0x8000 else val


def read_mpu9250(bus: SMBus):
    data = bus.read_i2c_block_data(MPU9250_ADDR, ACCEL_XOUT_H, 14)
    ax = _to_int16(data[0], data[1]) / ACCEL_SCALE
    ay = _to_int16(data[2], data[3]) / ACCEL_SCALE
    az = _to_int16(data[4], data[5]) / ACCEL_SCALE
    temp_raw = _to_int16(data[6], data[7])
    temp_c = temp_raw / 333.87 + 21.0
    gx = _to_int16(data[8], data[9]) / GYRO_SCALE
    gy = _to_int16(data[10], data[11]) / GYRO_SCALE
    gz = _to_int16(data[12], data[13]) / GYRO_SCALE
    return (ax, ay, az), (gx, gy, gz), temp_c


# ---------- BMP280 ----------
class BMP280:
    def __init__(self, bus: SMBus, addr: int = BMP280_ADDR) -> None:
        self.bus = bus
        self.addr = addr
        chip_id = bus.read_byte_data(addr, BMP280_ID_REG)
        print(f"[BMP280] CHIP_ID = 0x{chip_id:02X} (expected 0x58)")
        # normal mode, osrs_t=x1, osrs_p=x4
        bus.write_byte_data(addr, BMP280_CTRL_MEAS, (0b001 << 5) | (0b011 << 2) | 0b11)
        bus.write_byte_data(addr, BMP280_CONFIG, (0b100 << 2))
        self._load_calib()
        self.t_fine = 0

    def _load_calib(self) -> None:
        raw = self.bus.read_i2c_block_data(self.addr, BMP280_CALIB, 24)
        self.dig_T1, self.dig_T2, self.dig_T3 = struct.unpack("<Hhh", bytes(raw[0:6]))
        self.dig_P = struct.unpack("<Hhhhhhhhh", bytes(raw[6:24]))

    def read(self):
        d = self.bus.read_i2c_block_data(self.addr, BMP280_PRESS_MSB, 6)
        adc_p = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)
        adc_t = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)
        # temperature (Bosch reference formula)
        var1 = (adc_t / 16384.0 - self.dig_T1 / 1024.0) * self.dig_T2
        var2 = ((adc_t / 131072.0 - self.dig_T1 / 8192.0) ** 2) * self.dig_T3
        self.t_fine = var1 + var2
        temp_c = (var1 + var2) / 5120.0
        # pressure
        P1, P2, P3, P4, P5, P6, P7, P8, P9 = self.dig_P
        v1 = self.t_fine / 2.0 - 64000.0
        v2 = v1 * v1 * P6 / 32768.0
        v2 = v2 + v1 * P5 * 2.0
        v2 = v2 / 4.0 + P4 * 65536.0
        v1 = (P3 * v1 * v1 / 524288.0 + P2 * v1) / 524288.0
        v1 = (1.0 + v1 / 32768.0) * P1
        if v1 == 0:
            return temp_c, float("nan")
        p = 1048576.0 - adc_p
        p = (p - v2 / 4096.0) * 6250.0 / v1
        v1 = P9 * p * p / 2147483648.0
        v2 = p * P8 / 32768.0
        p = p + (v1 + v2 + P7) / 16.0
        return temp_c, p / 100.0  # hPa


def main() -> None:
    print(f"Opening /dev/i2c-{I2C_BUS}...")
    with SMBus(I2C_BUS) as bus:
        init_mpu9250(bus)

        bmp = None
        for addr in (BMP280_ADDR, 0x77):
            try:
                bmp = BMP280(bus, addr)
                break
            except OSError:
                continue
        if bmp is None:
            print("[BMP280] not detected at 0x76 or 0x77 — skipping barometer.")

        print("\nReading at ~20 Hz. Ctrl+C to stop.\n")
        header = f"{'ax':>7} {'ay':>7} {'az':>7} | {'gx':>8} {'gy':>8} {'gz':>8} | {'T_imu':>6}"
        if bmp is not None:
            header += f" | {'T_baro':>6} {'P_hPa':>8}"
        print(header)
        try:
            while True:
                acc, gyr, t_imu = read_mpu9250(bus)
                line = (f"{acc[0]:+7.3f} {acc[1]:+7.3f} {acc[2]:+7.3f} | "
                        f"{gyr[0]:+8.2f} {gyr[1]:+8.2f} {gyr[2]:+8.2f} | "
                        f"{t_imu:6.2f}")
                if bmp is not None:
                    t_baro, p_hpa = bmp.read()
                    line += f" | {t_baro:6.2f} {p_hpa:8.2f}"
                print(line)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
