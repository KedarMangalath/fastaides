# Generated by Django 5.1.3 on 2024-12-03 10:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0003_alter_chatsession_state'),
    ]

    operations = [
        migrations.AddField(
            model_name='medicineorder',
            name='metadata',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='message',
            name='metadata',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='prescription',
            name='validation_errors',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='chatsession',
            name='state',
            field=models.CharField(choices=[('CHAT', 'Chat'), ('PRESCRIPTION_REVIEW', 'Prescription Review'), ('ORDER_QUANTITY', 'Order Quantity'), ('COLLECT_INFO', 'Collect Info'), ('CONFIRM_ORDER', 'Confirm Order'), ('ORDER_COMPLETE', 'Order Complete'), ('MANUAL_REVIEW', 'Manual Review'), ('AWAITING_PRESCRIPTION', 'Awaiting Prescription')], default='CHAT', max_length=50),
        ),
    ]
