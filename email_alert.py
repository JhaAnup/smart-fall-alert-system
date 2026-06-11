import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import cv2
import os


class EmailAlert:
    # Handles SMTP email alerts with image attachments
    def __init__(self):
        self.sender_email = None
        self.sender_password = None
        self.receiver_email = None
        self.configured = False
    
    def configure(self, sender_email, sender_password, receiver_email):
        # Configure email settings
        # sender_email: Gmail address to send from
        # sender_password: App password
        # receiver_email: Email address to send alerts to
        
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        self.configured = True
    
    def send_alert(self, detection_frame, input_type, confidence, timestamp=None):
        # Send fall detection alert email
        # detection_frame: OpenCV image of the fall detection
        # input_type: Image, Video or Live Camera
        # confidence: Confidence score of the fall detection
        
        if not self.configured:
            return False, "Email not configured. Please set up email settings first."
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = "🚨 Emergency Alert: Fall Detected"
            
            # Email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 10px;">
                    <h1>⚠️ FALL DETECTION ALERT</h1>
                </div>
                
                <div style="padding: 20px; background-color: #f5f5f5; margin-top: 10px; border-radius: 10px;">
                    <h2>Detection Details:</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 10px; font-weight: bold;">Time of Detection:</td>
                            <td style="padding: 10px;">{timestamp}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 10px; font-weight: bold;">Input Source:</td>
                            <td style="padding: 10px;">{input_type}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 10px; font-weight: bold;">Confidence Score:</td>
                            <td style="padding: 10px;">{confidence:.2%}</td>
                        </tr>
                    </table>
                </div>
                
                <div style="padding: 20px; margin-top: 10px;">
                    <h3>📸 Detection Frame:</h3>
                    <p>Please see the attached image showing the detected fall.</p>
                </div>
                
                <div style="padding: 20px; background-color: #fff3cd; border-left: 4px solid #ffc107; margin-top: 20px;">
                    <p><strong>⚠️ Action Required:</strong></p>
                    <p>Immediate attention may be needed. Please check on the individual and provide assistance if necessary.</p>
                </div>
                
                <div style="padding: 20px; color: #666; font-size: 12px; margin-top: 20px; border-top: 1px solid #ddd;">
                    <p>This is an automated alert from the Fall Detection System.</p>
                    <p>Powered by Deep Learning CNN-LSTM Model</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach detection image
            # Save frame temporarily
            temp_image_path = f"fall_detected_{int(datetime.now().timestamp())}.jpg"
            cv2.imwrite(temp_image_path, detection_frame)
            
            with open(temp_image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(temp_image_path))
                msg.attach(image)
            
            # Send email via Gmail SMTP
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            # Clean up temp file
            os.remove(temp_image_path)
            
            return True, f"✅ Alert email sent successfully to {self.receiver_email}"
            
        except smtplib.SMTPAuthenticationError:
            return False, "❌ Authentication failed. Please check your email and app password."
        except smtplib.SMTPException as e:
            return False, f"❌ SMTP error: {str(e)}"
        except Exception as e:
            return False, f"❌ Error sending email: {str(e)}"
    
    def send_test_email(self):
        # Send a test email to verify configuration

        if not self.configured:
            return False, "Email not configured. Please set up email settings first."
        
        try:
            # Create simple test message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = "✅ Fall Detection System - Test Email"
            
            body = """
            <html>
            <body style="font-family: Arial, sans-serif;">
                <div style="background-color: #4CAF50; color: white; padding: 20px; border-radius: 10px;">
                    <h1>✅ Test Email Successful</h1>
                </div>
                
                <div style="padding: 20px; margin-top: 10px;">
                    <p>Your Fall Detection System email configuration is working correctly!</p>
                    <p>You will receive alerts at this email address when a fall is detected.</p>
                </div>
                
                <div style="padding: 20px; color: #666; font-size: 12px; margin-top: 20px; border-top: 1px solid #ddd;">
                    <p>This is a test message from the Fall Detection System.</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            return True, f"✅ Test email sent successfully to {self.receiver_email}"
            
        except smtplib.SMTPAuthenticationError:
            return False, "❌ Authentication failed. Please check your email and app password."
        except Exception as e:
            return False, f"❌ Error sending test email: {str(e)}"


# Global email alert instance
email_alert = EmailAlert()
