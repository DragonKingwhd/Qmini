import yaml
p = 'sim2real/config/calibration.yaml'
c = yaml.safe_load(open(p))
c['joints']['motor_zero_rad'][1] = -2.676
c['joints']['motor_zero_rad'][6] = -2.181
yaml.safe_dump(c, open(p, 'w'), sort_keys=False, allow_unicode=True)
print('hip_roll 零位已更新 →', c['joints']['motor_zero_rad'])
