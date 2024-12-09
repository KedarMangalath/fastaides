# Generated by Django 5.1.3 on 2024-11-28 04:07

import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ChatSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.CharField(default=uuid.uuid4, max_length=100, unique=True)),
                ('state', models.CharField(default='CHAT', max_length=50)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('last_context', models.JSONField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Message',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('role', models.CharField(max_length=20)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('type', models.CharField(default='text', max_length=20)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='messages', to='chat.chatsession')),
            ],
        ),
        migrations.CreateModel(
            name='Prescription',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='prescriptions/')),
                ('prescription_type', models.CharField(choices=[('DIGITAL', 'Digital'), ('HANDWRITTEN', 'Handwritten'), ('INVALID', 'Invalid')], max_length=20)),
                ('reference_number', models.CharField(default=uuid.uuid4, max_length=50, unique=True)),
                ('status', models.CharField(choices=[('PROCESSING', 'Processing'), ('MANUAL_REVIEW', 'Manual Review'), ('COMPLETED', 'Completed'), ('REJECTED', 'Rejected')], default='PROCESSING', max_length=20)),
                ('extracted_data', models.JSONField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='prescriptions', to='chat.chatsession')),
            ],
        ),
    ]
