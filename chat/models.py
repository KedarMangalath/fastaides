# models.py
from django.db import models
from django.utils import timezone
import uuid

class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True, default=uuid.uuid4)
    state = models.CharField(max_length=50, default='CHAT', choices=[
        ('CHAT', 'Chat'),
        ('PRESCRIPTION_REVIEW', 'Prescription Review'),
        ('ORDER_QUANTITY', 'Order Quantity'),
        ('COLLECT_INFO', 'Collect Info'),
        ('CONFIRM_ORDER', 'Confirm Order'),
        ('ORDER_COMPLETE', 'Order Complete'),
        ('MANUAL_REVIEW', 'Manual Review'),
        ('AWAITING_PRESCRIPTION', 'Awaiting Prescription')  # Added new state
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_context = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Chat Session {self.session_id}"
    
    def get_recent_messages(self, limit=10):
        """Get recent messages for this session"""
        return self.messages.order_by('-timestamp')[:limit]
    
    def get_prescription_context(self):
        """Get current valid prescription context"""
        try:
            latest_prescription = self.prescriptions.filter(
                status='COMPLETED'
            ).latest('created_at')
            
            if latest_prescription:
                return {
                    'prescription_id': latest_prescription.id,
                    'medicines': latest_prescription.extracted_data,
                    'prescription_type': latest_prescription.prescription_type,
                    'reference_number': latest_prescription.reference_number
                }
        except Prescription.DoesNotExist:
            pass
        return None
    
    def get_current_order_state(self):
        """Get current order state and details"""
        try:
            latest_order = self.orders.latest('created_at')
            if latest_order:
                return {
                    'order_id': latest_order.id,
                    'status': latest_order.status,
                    'order_details': latest_order.order_details
                }
        except MedicineOrder.DoesNotExist:
            pass
        return None

class Message(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    role = models.CharField(max_length=20)  # 'user' or 'assistant'
    timestamp = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=20, default='text')  # text, prescription, order
    metadata = models.JSONField(null=True, blank=True)  # Added for additional context

    def __str__(self):
        return f"{self.role} message in {self.session.session_id}"
    
    class Meta:
        ordering = ['timestamp']

class CustomerDetails(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='customer_details')
    name = models.CharField(max_length=100)
    phone = models.CharField(max_length=15)
    email = models.EmailField(blank=True, null=True)
    address = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Customer {self.name} ({self.phone})"

class Prescription(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='prescriptions')
    image = models.ImageField(upload_to='prescriptions/')
    prescription_type = models.CharField(max_length=20, choices=[
        ('DIGITAL', 'Digital'),
        ('HANDWRITTEN', 'Handwritten'),
        ('INVALID', 'Invalid')
    ])
    reference_number = models.CharField(max_length=50, unique=True, default=uuid.uuid4)
    status = models.CharField(max_length=20, choices=[
        ('PROCESSING', 'Processing'),
        ('MANUAL_REVIEW', 'Manual Review'),
        ('COMPLETED', 'Completed'),
        ('REJECTED', 'Rejected')
    ], default='PROCESSING')
    extracted_data = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    validation_errors = models.JSONField(null=True, blank=True)  # Added to store validation issues

    def __str__(self):
        return f"Prescription {self.reference_number}"

class MedicineOrder(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='orders')
    customer = models.ForeignKey(CustomerDetails, on_delete=models.CASCADE)
    prescription = models.ForeignKey(Prescription, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=[
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('CONFIRMED', 'Confirmed'),
        ('CANCELLED', 'Cancelled')
    ], default='PENDING')
    order_details = models.JSONField()  # Store medicine names, quantities, etc
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(null=True, blank=True)  # Added for order processing history

    def __str__(self):
        return f"Order {self.id} for {self.customer.name}"
    
    class Meta:
        ordering = ['-created_at']