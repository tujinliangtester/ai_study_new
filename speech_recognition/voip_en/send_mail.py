import smtplib
from email.mime.text import MIMEText
from email.header import Header


def send_mail(s):
    # 第三方 SMTP 服务
    mail_host = "smtp.163.com"  # 设置服务器
    mail_user = "tujinliangtester"  # 用户名
    mail_pass = "6745425tjl"  # 口令

    sender = 'tujinliangtester@163.com'
    receivers = ['tujinliangtester@163.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    message = MIMEText(s, 'plain', 'utf-8')
    message['From'] = Header("涂金良", 'utf-8')
    message['To'] = Header("涂金良", 'utf-8')

    subject = 'Python SMTP 邮件测试'
    message['Subject'] = Header(subject, 'utf-8')

    try:
        # smtpObj = smtplib.SMTP()
        # smtpObj.connect(mail_host, 25)  # 25 为 SMTP 端口号
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)

        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")

    except smtplib.SMTPException as e:
        print("Error: 无法发送邮件")
        print(e)
#
# s='\xc7\xeb\xca\xb9\xd3\xc3\xca\xda\xc8\xa8\xc2\xeb\xb5\xc7\xc2\xbc\xa1\xa3\xcf\xea\xc7\xe9\xc7\xeb\xbf\xb4'
# print(s.encode('gbk'))