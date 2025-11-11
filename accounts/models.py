from mongoengine import Document, EmbeddedDocument, fields
from django.contrib.auth.hashers import make_password, check_password
from datetime import datetime
from decimal import Decimal


class TradingUser(Document):
    """Primary trading user - designed for single user initially but scalable"""
    username = fields.StringField(max_length=150, unique=True, required=True)
    email = fields.EmailField(unique=True, required=True)
    password = fields.StringField(max_length=128, required=True)
    is_active = fields.BooleanField(default=True)
    is_primary_trader = fields.BooleanField(default=True)  # For single user setup
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'trading_users',
        'indexes': ['email', 'username', 'is_primary_trader']
    }

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        return check_password(raw_password, self.password)

    def save(self, *args, **kwargs):
        self.updated_at = datetime.utcnow()
        super().save(*args, **kwargs)

    @classmethod
    def get_primary_trader(cls):
        """Get the primary trader for single-user setup"""
        return cls.objects(is_primary_trader=True).first()

    def __str__(self):
        return f"Trader: {self.username}"
