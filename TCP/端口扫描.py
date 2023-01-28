from scapy.layers.inet import IP, TCP, ICMP,UDP
from scapy.sendrecv import sr1


def print_ports(port, state):
    print("%s | %s" % (port, state))



def udpScan(target,ports):
    print("UDP 扫描 %s with ports %s" % (target, ports))
    for port in ports:
        udp_scan_resp = sr1(IP(dst=target)/UDP(dport=port),timeout=5)
        if (str(type(udp_scan_resp))=="<class 'NoneType'>"):
            print_ports(port, "Open|Filtered")
        elif(udp_scan_resp.haslayer(UDP)):
            if(udp_scan_resp.getlayer(TCP).flags == "R"):
                print_ports(port, "Open")
        elif(udp_scan_resp.haslayer(ICMP)):
            if(int(udp_scan_resp.getlayer(ICMP).type)==3 and int(udp_scan_resp.getlayer(ICMP).code) in [1,2,3,9,10,13]):
                print_ports(port, "Filtered")

def synScan(target, ports):
    print("tcp全连接扫描 %s with ports %s" % (target, ports))
    for port in ports:
        send = sr1(IP(dst=target)/TCP(dport=port, flags="S"),
                   timeout=2, verbose=0)
        if (send is None):
            print_ports(port, "closed")
        elif send.haslayer("TCP"):
            print(send["TCP"].flags)
            if send["TCP"].flags == "SA":
                send_1 = sr1(IP(dst=target) / TCP(dport=port,
                             flags="R"), timeout=2, verbose=0)  # 只修改这里
                print_ports(port, "opend")
            elif send["TCP"].flags == "RA":
                print_ports(port, "closed")


def nullScan(target, port: int):
    print("tcp NULL 扫描 %s with ports %s" % (target, port))
    null_scan_resp = sr1(
        IP(dst=target)/TCP(dport=port, flags=""), timeout=5)
    if (str(type(null_scan_resp)) == "<class 'NoneType'>"):
        print_ports(port, "Open|Filtered")
    elif (null_scan_resp.haslayer(TCP)):
        if (null_scan_resp.getlayer(TCP).flags == "R" or null_scan_resp.getlayer(TCP).flags == "A"):
            print_ports(port, "Closed")
    elif (null_scan_resp.haslayer(ICMP)):
        if (int(null_scan_resp.getlayer(ICMP).type) == 3 and int(null_scan_resp.getlayer(ICMP).code) in [1, 2, 3, 9, 10, 13]):
            print_ports(port, "Filtered")


def ackScan(target, ports):
    print("tcp ack扫描 %s with ports %s" % (target, ports))
    for port in ports:
        ack_flag_scan_resp = sr1(
            IP(dst=target)/TCP(dport=port, flags="A"), timeout=5)
        print(str(type(ack_flag_scan_resp)))
        if (str(type(ack_flag_scan_resp)) == "<class 'NoneType'>"):
            print_ports(port, "filtered")
        elif (ack_flag_scan_resp.haslayer(TCP)):
            if (ack_flag_scan_resp.getlayer(TCP).flags == "R"):
                print_ports(port, "unfiltered")
        elif (ack_flag_scan_resp.haslayer(ICMP)):
            if (int(ack_flag_scan_resp.getlayer(ICMP).type) == 3 and int(ack_flag_scan_resp.getlayer(ICMP).code) in [1, 2, 3, 9, 10, 13]):
                print_ports(port, "filtered")
        else:
            print_ports(port, "filtered")

def finScan(target,ports):
    print("tcp FIN 扫描 %s with ports %s" % (target, ports))
    for port in ports:
        fin_scan_resp = sr1(IP(dst=target)/TCP(dport=port,flags="F"),timeout=5)
        if (str(type(fin_scan_resp))=="<class 'NoneType'>"):
            print_ports(port, "Open|Filtered")
        elif(fin_scan_resp.haslayer(TCP)):
            if(fin_scan_resp.getlayer(TCP).flags == 0x14):
                print_ports(port, "Closed")
        elif(fin_scan_resp.haslayer(ICMP)):
            if(int(fin_scan_resp.getlayer(ICMP).type)==3 and int(fin_scan_resp.getlayer(ICMP).code) in [1,2,3,9,10,13]):
                print_ports(port, "Filtered")


def tcpScan(target, port):
    send = sr1(IP(dst=target)/TCP(dport=port, flags="S"),timeout=0.5, verbose=0)
    if (send is None):
        print_ports(port, "closed")
        pass
    elif send.haslayer("TCP"):

        if send["TCP"].flags == "SA":
            #send_1 = sr1(IP(dst=target) / TCP(dport=port,flags="AR"), timeout=2, verbose=0)
            print_ports(port, "open")
        elif send["TCP"].flags == "RA":
            print_ports(port, "close")

for i in range(24445,24446):
    tcpScan('43.138.161.163',i)