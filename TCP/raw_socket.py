import socket 
from impacket import ImpactPacket

s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IPV4)
ip = ImpactPacket.IP()
tcp = ImpactPacket.TCP()
ip.set_ip_src("192.168.2.2")#你的ip
ip.set_ip_dst("113.100.155.155")#目标ip
ip.set_ip_ttl(255)#ttl
ip.set_ip_p(ImpactPacket.TCP.protocol)
tcp.set_th_flags(0b000010)#将syn标志位设为1
tcp.set_th_sport(12228)#源端口
tcp.set_th_dport(81)#目标端口
tcp.set_th_ack(0)
tcp.set_th_seq(0)
tcp.set_th_win(20000)#设置Window Size
ip.contains(tcp)
ip.calculate_checksum()
s.sendto(ip.get_packet(),('113.100.155.155',81))
b = s.recv(1024)
print(b.decode())
