"""
Enterprise Alert & Notification System - Система алертов и уведомлений
Обеспечивает многоканальные уведомления с интеллектуальной маршрутизацией и эскалацией
"""

import asyncio
import json
import time
import aiohttp
import smtplib
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import jinja2
from prometheus_client import Counter, Histogram, Gauge
import requests
from twilio.rest import Client as TwilioClient
import telegram
from slack_sdk.webhook import WebhookClient
import discord
from collections import defaultdict, deque

class NotificationChannel(Enum):
    """Каналы уведомлений"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    PUSH = "push"
    VOICE = "voice"

class AlertSeverity(Enum):
    """Уровни критичности алертов"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationStatus(Enum):
    """Статусы уведомлений"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRY = "retry"

class EscalationLevel(Enum):
    """Уровни эскалации"""
    L1 = "l1"  # Первая линия поддержки
    L2 = "l2"  # Вторая линия поддержки
    L3 = "l3"  # Старшие инженеры
    MANAGEMENT = "management"  # Менеджмент

@dataclass
class NotificationRule:
    """Правило уведомления"""
    id: str
    name: str
    conditions: Dict[str, Any]  # Условия срабатывания
    channels: List[NotificationChannel]
    recipients: List[str]
    template: str
    priority: int = 1
    enabled: bool = True
    rate_limit: Optional[int] = None  # Максимум уведомлений в час
    quiet_hours: Optional[Dict[str, str]] = None  # Часы тишины

@dataclass
class EscalationRule:
    """Правило эскалации"""
    id: str
    alert_type: str
    severity: AlertSeverity
    levels: List[Dict[str, Any]]  # Уровни эскалации с таймаутами
    enabled: bool = True

@dataclass
class NotificationTemplate:
    """Шаблон уведомления"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    variables: List[str]

@dataclass
class Notification:
    """Уведомление"""
    id: str
    alert_id: str
    channel: NotificationChannel
    recipient: str
    subject: str
    content: str
    status: NotificationStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Recipient:
    """Получатель уведомлений"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_user_id: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.L1
    timezone: str = "UTC"
    quiet_hours: Optional[Dict[str, str]] = None
    preferred_channels: List[NotificationChannel] = None

# Метрики
NOTIFICATIONS_SENT = Counter('notifications_sent_total', 'Total notifications sent', ['channel', 'status'])
NOTIFICATION_DELIVERY_TIME = Histogram('notification_delivery_time_seconds', 'Notification delivery time', ['channel'])
ESCALATIONS_TRIGGERED = Counter('escalations_triggered_total', 'Total escalations triggered', ['level'])
NOTIFICATION_FAILURES = Counter('notification_failures_total', 'Notification failures', ['channel', 'error_type'])

class EnterpriseAlertNotificationSystem:
    """Enterprise система алертов и уведомлений"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Состояние системы
        self.notification_rules: Dict[str, NotificationRule] = {}
        self.escalation_rules: Dict[str, EscalationRule] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.recipients: Dict[str, Recipient] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Очереди уведомлений
        self.notification_queues: Dict[NotificationChannel, deque] = {
            channel: deque() for channel in NotificationChannel
        }
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Клиенты для различных каналов
        self.notification_clients = {}
        self._init_notification_clients()
        
        # Jinja2 для шаблонов
        self.template_env = jinja2.Environment(
            loader=jinja2.DictLoader({}),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Инициализация правил и шаблонов
        self._init_default_rules()
        self._init_default_templates()
        self._init_default_recipients()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_alert_notification')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _init_notification_clients(self):
        """Инициализация клиентов уведомлений"""
        # Twilio для SMS
        if self.config.get('twilio_account_sid'):
            self.notification_clients[NotificationChannel.SMS] = TwilioClient(
                self.config['twilio_account_sid'],
                self.config['twilio_auth_token']
            )
            
        # Telegram Bot
        if self.config.get('telegram_bot_token'):
            self.notification_clients[NotificationChannel.TELEGRAM] = telegram.Bot(
                token=self.config['telegram_bot_token']
            )
            
        # Slack Webhook
        if self.config.get('slack_webhook_url'):
            self.notification_clients[NotificationChannel.SLACK] = WebhookClient(
                url=self.config['slack_webhook_url']
            )
            
        # Discord Webhook
        if self.config.get('discord_webhook_url'):
            self.notification_clients[NotificationChannel.DISCORD] = self.config['discord_webhook_url']
            
    def _init_default_rules(self):
        """Инициализация правил по умолчанию"""
        # Правила уведомлений
        self.notification_rules = {
            'critical_system_alert': NotificationRule(
                id='critical_system_alert',
                name='Critical System Alert',
                conditions={
                    'severity': ['critical', 'emergency'],
                    'component': ['trading_engine', 'database', 'api_gateway']
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK],
                recipients=['admin', 'devops_team'],
                template='critical_alert',
                priority=1
            ),
            'trading_alert': NotificationRule(
                id='trading_alert',
                name='Trading Alert',
                conditions={
                    'component': ['trading_engine'],
                    'metric_name': ['trading_latency', 'trading_pnl']
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.TELEGRAM],
                recipients=['trading_team', 'risk_manager'],
                template='trading_alert',
                priority=2
            ),
            'performance_warning': NotificationRule(
                id='performance_warning',
                name='Performance Warning',
                conditions={
                    'severity': ['warning'],
                    'metric_name': ['cpu_usage', 'memory_usage', 'disk_usage']
                },
                channels=[NotificationChannel.SLACK],
                recipients=['devops_team'],
                template='performance_warning',
                priority=3,
                rate_limit=5  # Максимум 5 уведомлений в час
            )
        }
        
        # Правила эскалации
        self.escalation_rules = {
            'critical_escalation': EscalationRule(
                id='critical_escalation',
                alert_type='system',
                severity=AlertSeverity.CRITICAL,
                levels=[
                    {'level': EscalationLevel.L1, 'timeout_minutes': 5, 'recipients': ['devops_team']},
                    {'level': EscalationLevel.L2, 'timeout_minutes': 15, 'recipients': ['senior_engineers']},
                    {'level': EscalationLevel.L3, 'timeout_minutes': 30, 'recipients': ['tech_lead']},
                    {'level': EscalationLevel.MANAGEMENT, 'timeout_minutes': 60, 'recipients': ['cto']}
                ]
            ),
            'emergency_escalation': EscalationRule(
                id='emergency_escalation',
                alert_type='system',
                severity=AlertSeverity.EMERGENCY,
                levels=[
                    {'level': EscalationLevel.L1, 'timeout_minutes': 2, 'recipients': ['devops_team', 'senior_engineers']},
                    {'level': EscalationLevel.L2, 'timeout_minutes': 5, 'recipients': ['tech_lead', 'cto']},
                    {'level': EscalationLevel.MANAGEMENT, 'timeout_minutes': 10, 'recipients': ['ceo']}
                ]
            )
        }
        
    def _init_default_templates(self):
        """Инициализация шаблонов по умолчанию"""
        self.templates = {
            'critical_alert': NotificationTemplate(
                id='critical_alert',
                name='Critical Alert Template',
                channel=NotificationChannel.EMAIL,
                subject_template='🚨 CRITICAL ALERT: {{ alert.title }}',
                body_template='''
                <h2>Critical Alert Detected</h2>
                <p><strong>Title:</strong> {{ alert.title }}</p>
                <p><strong>Description:</strong> {{ alert.description }}</p>
                <p><strong>Component:</strong> {{ alert.component }}</p>
                <p><strong>Severity:</strong> {{ alert.severity }}</p>
                <p><strong>Current Value:</strong> {{ alert.current_value }}</p>
                <p><strong>Threshold:</strong> {{ alert.threshold_value }}</p>
                <p><strong>Timestamp:</strong> {{ alert.timestamp }}</p>
                
                <h3>Immediate Actions Required:</h3>
                <ul>
                    <li>Check system status dashboard</li>
                    <li>Review recent deployments</li>
                    <li>Escalate if not resolved within 15 minutes</li>
                </ul>
                ''',
                variables=['alert']
            ),
            'trading_alert': NotificationTemplate(
                id='trading_alert',
                name='Trading Alert Template',
                channel=NotificationChannel.TELEGRAM,
                subject_template='📈 Trading Alert: {{ alert.title }}',
                body_template='''
                🔔 *Trading Alert*
                
                *Title:* {{ alert.title }}
                *Component:* {{ alert.component }}
                *Metric:* {{ alert.metric_name }}
                *Current Value:* {{ alert.current_value }}
                *Threshold:* {{ alert.threshold_value }}
                *Time:* {{ alert.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                
                Please review trading parameters and take necessary actions.
                ''',
                variables=['alert']
            ),
            'performance_warning': NotificationTemplate(
                id='performance_warning',
                name='Performance Warning Template',
                channel=NotificationChannel.SLACK,
                subject_template='⚠️ Performance Warning: {{ alert.title }}',
                body_template='''
                {
                    "text": "Performance Warning Detected",
                    "attachments": [
                        {
                            "color": "warning",
                            "title": "{{ alert.title }}",
                            "fields": [
                                {"title": "Component", "value": "{{ alert.component }}", "short": true},
                                {"title": "Metric", "value": "{{ alert.metric_name }}", "short": true},
                                {"title": "Current Value", "value": "{{ alert.current_value }}", "short": true},
                                {"title": "Threshold", "value": "{{ alert.threshold_value }}", "short": true}
                            ],
                            "timestamp": "{{ alert.timestamp.isoformat() }}"
                        }
                    ]
                }
                ''',
                variables=['alert']
            )
        }
        
    def _init_default_recipients(self):
        """Инициализация получателей по умолчанию"""
        self.recipients = {
            'admin': Recipient(
                id='admin',
                name='System Administrator',
                email='admin@peper-binance.com',
                phone='+1234567890',
                slack_user_id='U123456789',
                telegram_chat_id='123456789',
                escalation_level=EscalationLevel.L1,
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
            ),
            'devops_team': Recipient(
                id='devops_team',
                name='DevOps Team',
                email='devops@peper-binance.com',
                slack_user_id='C123456789',  # Channel ID
                escalation_level=EscalationLevel.L1,
                preferred_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            'trading_team': Recipient(
                id='trading_team',
                name='Trading Team',
                email='trading@peper-binance.com',
                telegram_chat_id='-123456789',  # Group chat ID
                escalation_level=EscalationLevel.L2,
                preferred_channels=[NotificationChannel.TELEGRAM, NotificationChannel.EMAIL]
            ),
            'risk_manager': Recipient(
                id='risk_manager',
                name='Risk Manager',
                email='risk@peper-binance.com',
                phone='+1234567891',
                escalation_level=EscalationLevel.L2,
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
            ),
            'senior_engineers': Recipient(
                id='senior_engineers',
                name='Senior Engineers',
                email='senior-eng@peper-binance.com',
                slack_user_id='C987654321',
                escalation_level=EscalationLevel.L2,
                preferred_channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL]
            ),
            'tech_lead': Recipient(
                id='tech_lead',
                name='Technical Lead',
                email='tech-lead@peper-binance.com',
                phone='+1234567892',
                escalation_level=EscalationLevel.L3,
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
            ),
            'cto': Recipient(
                id='cto',
                name='Chief Technology Officer',
                email='cto@peper-binance.com',
                phone='+1234567893',
                escalation_level=EscalationLevel.MANAGEMENT,
                preferred_channels=[NotificationChannel.EMAIL, NotificationChannel.SMS]
            )
        }
        
    async def start(self):
        """Запуск системы уведомлений"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Запуск обработчиков уведомлений
        for channel in NotificationChannel:
            asyncio.create_task(self._notification_processor(channel))
            
        # Запуск системы эскалации
        asyncio.create_task(self._escalation_processor())
        
        # Запуск мониторинга доставки
        asyncio.create_task(self._delivery_monitor())
        
        # Запуск очистки старых уведомлений
        asyncio.create_task(self._cleanup_old_notifications())
        
        self.logger.info("Enterprise Alert & Notification System started")
        
    async def stop(self):
        """Остановка системы"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def process_alert(self, alert: Dict[str, Any]):
        """Обработка алерта"""
        try:
            # Поиск подходящих правил уведомлений
            matching_rules = self._find_matching_rules(alert)
            
            for rule in matching_rules:
                if not rule.enabled:
                    continue
                    
                # Проверка rate limiting
                if not self._check_rate_limit(rule):
                    continue
                    
                # Создание уведомлений
                await self._create_notifications_for_rule(alert, rule)
                
            # Запуск эскалации для критических алертов
            if alert.get('severity') in ['critical', 'emergency']:
                await self._start_escalation(alert)
                
        except Exception as e:
            self.logger.error(f"Alert processing error: {e}")
            
    def _find_matching_rules(self, alert: Dict[str, Any]) -> List[NotificationRule]:
        """Поиск подходящих правил уведомлений"""
        matching_rules = []
        
        for rule in self.notification_rules.values():
            if self._rule_matches_alert(rule, alert):
                matching_rules.append(rule)
                
        # Сортировка по приоритету
        matching_rules.sort(key=lambda r: r.priority)
        
        return matching_rules
        
    def _rule_matches_alert(self, rule: NotificationRule, alert: Dict[str, Any]) -> bool:
        """Проверка соответствия правила алерту"""
        for condition_key, condition_values in rule.conditions.items():
            alert_value = alert.get(condition_key)
            
            if alert_value is None:
                return False
                
            if isinstance(condition_values, list):
                if alert_value not in condition_values:
                    return False
            else:
                if alert_value != condition_values:
                    return False
                    
        return True
        
    def _check_rate_limit(self, rule: NotificationRule) -> bool:
        """Проверка rate limiting"""
        if not rule.rate_limit:
            return True
            
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Очистка старых записей
        rule_key = f"rate_limit:{rule.id}"
        while self.rate_limits[rule_key] and self.rate_limits[rule_key][0] < hour_ago:
            self.rate_limits[rule_key].popleft()
            
        # Проверка лимита
        if len(self.rate_limits[rule_key]) >= rule.rate_limit:
            return False
            
        # Добавление новой записи
        self.rate_limits[rule_key].append(current_time)
        return True
        
    async def _create_notifications_for_rule(self, alert: Dict[str, Any], rule: NotificationRule):
        """Создание уведомлений для правила"""
        for recipient_id in rule.recipients:
            recipient = self.recipients.get(recipient_id)
            if not recipient:
                continue
                
            # Проверка часов тишины
            if self._is_quiet_hours(recipient):
                continue
                
            # Выбор каналов уведомлений
            channels = self._select_channels_for_recipient(rule.channels, recipient)
            
            for channel in channels:
                await self._create_notification(alert, rule, recipient, channel)
                
    def _is_quiet_hours(self, recipient: Recipient) -> bool:
        """Проверка часов тишины"""
        if not recipient.quiet_hours:
            return False
            
        # Простая проверка (в реальной системе нужно учитывать timezone)
        current_hour = datetime.now().hour
        start_hour = int(recipient.quiet_hours.get('start', '0'))
        end_hour = int(recipient.quiet_hours.get('end', '0'))
        
        if start_hour <= end_hour:
            return start_hour <= current_hour <= end_hour
        else:
            return current_hour >= start_hour or current_hour <= end_hour
            
    def _select_channels_for_recipient(self, rule_channels: List[NotificationChannel], 
                                     recipient: Recipient) -> List[NotificationChannel]:
        """Выбор каналов для получателя"""
        if recipient.preferred_channels:
            # Пересечение каналов правила и предпочтений получателя
            return [ch for ch in rule_channels if ch in recipient.preferred_channels]
        else:
            return rule_channels
            
    async def _create_notification(self, alert: Dict[str, Any], rule: NotificationRule, 
                                 recipient: Recipient, channel: NotificationChannel):
        """Создание уведомления"""
        try:
            # Получение шаблона
            template = self.templates.get(rule.template)
            if not template:
                self.logger.error(f"Template not found: {rule.template}")
                return
                
            # Рендеринг шаблона
            subject, content = await self._render_template(template, alert, recipient)
            
            # Создание уведомления
            notification = Notification(
                id=f"{alert['id']}_{recipient.id}_{channel.value}_{int(time.time())}",
                alert_id=alert['id'],
                channel=channel,
                recipient=self._get_recipient_address(recipient, channel),
                subject=subject,
                content=content,
                status=NotificationStatus.PENDING,
                created_at=datetime.now()
            )
            
            self.notifications[notification.id] = notification
            
            # Добавление в очередь
            self.notification_queues[channel].append(notification)
            
            # Сохранение в Redis
            await self._save_notification(notification)
            
            self.logger.info(f"Notification created: {notification.id}")
            
        except Exception as e:
            self.logger.error(f"Notification creation error: {e}")
            
    async def _render_template(self, template: NotificationTemplate, alert: Dict[str, Any], 
                             recipient: Recipient) -> tuple[str, str]:
        """Рендеринг шаблона"""
        context = {
            'alert': alert,
            'recipient': recipient,
            'timestamp': datetime.now()
        }
        
        # Рендеринг заголовка
        subject_template = self.template_env.from_string(template.subject_template)
        subject = subject_template.render(context)
        
        # Рендеринг содержимого
        body_template = self.template_env.from_string(template.body_template)
        content = body_template.render(context)
        
        return subject, content
        
    def _get_recipient_address(self, recipient: Recipient, channel: NotificationChannel) -> str:
        """Получение адреса получателя для канала"""
        if channel == NotificationChannel.EMAIL:
            return recipient.email
        elif channel == NotificationChannel.SMS:
            return recipient.phone
        elif channel == NotificationChannel.SLACK:
            return recipient.slack_user_id
        elif channel == NotificationChannel.TELEGRAM:
            return recipient.telegram_chat_id
        elif channel == NotificationChannel.DISCORD:
            return recipient.discord_user_id
        else:
            return recipient.id
            
    async def _notification_processor(self, channel: NotificationChannel):
        """Обработчик уведомлений для канала"""
        while True:
            try:
                queue = self.notification_queues[channel]
                
                if queue:
                    notification = queue.popleft()
                    await self._send_notification(notification)
                else:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Notification processor error for {channel}: {e}")
                await asyncio.sleep(5)
                
    async def _send_notification(self, notification: Notification):
        """Отправка уведомления"""
        try:
            notification.status = NotificationStatus.PENDING
            start_time = time.time()
            
            success = False
            
            if notification.channel == NotificationChannel.EMAIL:
                success = await self._send_email(notification)
            elif notification.channel == NotificationChannel.SMS:
                success = await self._send_sms(notification)
            elif notification.channel == NotificationChannel.SLACK:
                success = await self._send_slack(notification)
            elif notification.channel == NotificationChannel.TELEGRAM:
                success = await self._send_telegram(notification)
            elif notification.channel == NotificationChannel.DISCORD:
                success = await self._send_discord(notification)
            elif notification.channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(notification)
                
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                
                # Метрики
                delivery_time = time.time() - start_time
                NOTIFICATION_DELIVERY_TIME.labels(channel=notification.channel.value).observe(delivery_time)
                NOTIFICATIONS_SENT.labels(channel=notification.channel.value, status='success').inc()
                
                self.logger.info(f"Notification sent: {notification.id}")
            else:
                await self._handle_notification_failure(notification)
                
        except Exception as e:
            notification.error_message = str(e)
            await self._handle_notification_failure(notification)
            
        # Обновление в Redis
        await self._save_notification(notification)
        
    async def _send_email(self, notification: Notification) -> bool:
        """Отправка email уведомления"""
        try:
            smtp_config = self.config.get('smtp', {})
            if not smtp_config:
                return False
                
            msg = MimeMultipart()
            msg['From'] = smtp_config['from']
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            msg.attach(MimeText(notification.content, 'html'))
            
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
                
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email sending failed: {e}")
            return False
            
    async def _send_sms(self, notification: Notification) -> bool:
        """Отправка SMS уведомления"""
        try:
            twilio_client = self.notification_clients.get(NotificationChannel.SMS)
            if not twilio_client:
                return False
                
            message = twilio_client.messages.create(
                body=notification.content,
                from_=self.config['twilio_phone_number'],
                to=notification.recipient
            )
            
            return message.sid is not None
            
        except Exception as e:
            self.logger.error(f"SMS sending failed: {e}")
            return False
            
    async def _send_slack(self, notification: Notification) -> bool:
        """Отправка Slack уведомления"""
        try:
            slack_client = self.notification_clients.get(NotificationChannel.SLACK)
            if not slack_client:
                return False
                
            # Парсинг JSON контента для Slack
            try:
                slack_payload = json.loads(notification.content)
            except json.JSONDecodeError:
                slack_payload = {'text': notification.content}
                
            response = slack_client.send(**slack_payload)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack sending failed: {e}")
            return False
            
    async def _send_telegram(self, notification: Notification) -> bool:
        """Отправка Telegram уведомления"""
        try:
            telegram_bot = self.notification_clients.get(NotificationChannel.TELEGRAM)
            if not telegram_bot:
                return False
                
            await telegram_bot.send_message(
                chat_id=notification.recipient,
                text=notification.content,
                parse_mode='Markdown'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Telegram sending failed: {e}")
            return False
            
    async def _send_discord(self, notification: Notification) -> bool:
        """Отправка Discord уведомления"""
        try:
            webhook_url = self.notification_clients.get(NotificationChannel.DISCORD)
            if not webhook_url:
                return False
                
            payload = {
                'content': notification.content,
                'username': 'Peper Binance Alert Bot'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 204
                    
        except Exception as e:
            self.logger.error(f"Discord sending failed: {e}")
            return False
            
    async def _send_webhook(self, notification: Notification) -> bool:
        """Отправка webhook уведомления"""
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                return False
                
            payload = {
                'notification_id': notification.id,
                'alert_id': notification.alert_id,
                'subject': notification.subject,
                'content': notification.content,
                'timestamp': notification.created_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Webhook sending failed: {e}")
            return False
            
    async def _handle_notification_failure(self, notification: Notification):
        """Обработка неудачной отправки уведомления"""
        notification.retry_count += 1
        
        if notification.retry_count <= notification.max_retries:
            notification.status = NotificationStatus.RETRY
            
            # Экспоненциальная задержка
            delay = min(300, 2 ** notification.retry_count)  # Максимум 5 минут
            
            # Повторная постановка в очередь с задержкой
            asyncio.create_task(self._retry_notification(notification, delay))
            
            self.logger.warning(f"Notification retry scheduled: {notification.id}, attempt {notification.retry_count}")
        else:
            notification.status = NotificationStatus.FAILED
            
            NOTIFICATION_FAILURES.labels(
                channel=notification.channel.value,
                error_type='max_retries_exceeded'
            ).inc()
            
            self.logger.error(f"Notification failed permanently: {notification.id}")
            
    async def _retry_notification(self, notification: Notification, delay: int):
        """Повторная отправка уведомления с задержкой"""
        await asyncio.sleep(delay)
        self.notification_queues[notification.channel].append(notification)
        
    async def _start_escalation(self, alert: Dict[str, Any]):
        """Запуск эскалации"""
        escalation_rule = self._find_escalation_rule(alert)
        if not escalation_rule:
            return
            
        escalation_id = f"escalation_{alert['id']}_{int(time.time())}"
        
        # Сохранение информации об эскалации
        escalation_data = {
            'id': escalation_id,
            'alert_id': alert['id'],
            'rule_id': escalation_rule.id,
            'current_level': 0,
            'started_at': datetime.now().isoformat(),
            'resolved': False
        }
        
        await self.redis_client.setex(
            f"escalation:{escalation_id}",
            86400,  # 24 часа
            json.dumps(escalation_data)
        )
        
        # Запуск первого уровня эскалации
        await self._escalate_to_level(escalation_data, escalation_rule, 0)
        
    def _find_escalation_rule(self, alert: Dict[str, Any]) -> Optional[EscalationRule]:
        """Поиск правила эскалации"""
        for rule in self.escalation_rules.values():
            if (rule.enabled and 
                rule.severity.value == alert.get('severity') and
                alert.get('component') in ['trading_engine', 'database', 'api_gateway']):
                return rule
        return None
        
    async def _escalate_to_level(self, escalation_data: Dict, rule: EscalationRule, level_index: int):
        """Эскалация на уровень"""
        if level_index >= len(rule.levels):
            return
            
        level = rule.levels[level_index]
        
        ESCALATIONS_TRIGGERED.labels(level=level['level'].value).inc()
        
        self.logger.warning(f"Escalating to level {level['level'].value} for alert {escalation_data['alert_id']}")
        
        # Отправка уведомлений на текущий уровень
        # (здесь должна быть логика отправки уведомлений)
        
        # Планирование следующего уровня
        if level_index + 1 < len(rule.levels):
            timeout_seconds = level['timeout_minutes'] * 60
            asyncio.create_task(
                self._schedule_next_escalation(escalation_data, rule, level_index + 1, timeout_seconds)
            )
            
    async def _schedule_next_escalation(self, escalation_data: Dict, rule: EscalationRule, 
                                      next_level_index: int, delay_seconds: int):
        """Планирование следующего уровня эскалации"""
        await asyncio.sleep(delay_seconds)
        
        # Проверка, не разрешился ли алерт
        escalation_key = f"escalation:{escalation_data['id']}"
        current_data = await self.redis_client.get(escalation_key)
        
        if current_data:
            escalation_info = json.loads(current_data)
            if not escalation_info.get('resolved', False):
                await self._escalate_to_level(escalation_data, rule, next_level_index)
                
    async def _escalation_processor(self):
        """Обработчик эскалаций"""
        while True:
            try:
                # Проверка активных эскалаций
                escalation_keys = await self.redis_client.keys("escalation:*")
                
                for key in escalation_keys:
                    escalation_data = await self.redis_client.get(key)
                    if escalation_data:
                        escalation_info = json.loads(escalation_data)
                        
                        # Проверка, нужно ли разрешить эскалацию
                        if await self._should_resolve_escalation(escalation_info):
                            escalation_info['resolved'] = True
                            escalation_info['resolved_at'] = datetime.now().isoformat()
                            
                            await self.redis_client.setex(
                                key,
                                86400,
                                json.dumps(escalation_info)
                            )
                            
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Escalation processor error: {e}")
                await asyncio.sleep(30)
                
    async def _should_resolve_escalation(self, escalation_info: Dict) -> bool:
        """Проверка необходимости разрешения эскалации"""
        # Проверка, разрешился ли исходный алерт
        alert_key = f"alert:{escalation_info['alert_id']}"
        alert_data = await self.redis_client.get(alert_key)
        
        if alert_data:
            alert_info = json.loads(alert_data)
            return alert_info.get('resolved', False)
            
        return False
        
    async def _delivery_monitor(self):
        """Мониторинг доставки уведомлений"""
        while True:
            try:
                # Проверка статуса доставки уведомлений
                for notification in list(self.notifications.values()):
                    if notification.status == NotificationStatus.SENT:
                        # Проверка подтверждения доставки (если поддерживается каналом)
                        if await self._check_delivery_confirmation(notification):
                            notification.status = NotificationStatus.DELIVERED
                            notification.delivered_at = datetime.now()
                            
                            await self._save_notification(notification)
                            
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Delivery monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _check_delivery_confirmation(self, notification: Notification) -> bool:
        """Проверка подтверждения доставки"""
        # Здесь должна быть логика проверки доставки для каждого канала
        # Например, для SMS можно проверить delivery receipt
        return True
        
    async def _cleanup_old_notifications(self):
        """Очистка старых уведомлений"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(days=30)
                
                # Очистка из памяти
                to_remove = []
                for notification_id, notification in self.notifications.items():
                    if notification.created_at < cutoff_time:
                        to_remove.append(notification_id)
                        
                for notification_id in to_remove:
                    del self.notifications[notification_id]
                    
                # Очистка из Redis
                notification_keys = await self.redis_client.keys("notification:*")
                for key in notification_keys:
                    notification_data = await self.redis_client.get(key)
                    if notification_data:
                        notification_info = json.loads(notification_data)
                        created_at = datetime.fromisoformat(notification_info['created_at'])
                        
                        if created_at < cutoff_time:
                            await self.redis_client.delete(key)
                            
                await asyncio.sleep(86400)  # Очистка раз в день
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
                
    async def _save_notification(self, notification: Notification):
        """Сохранение уведомления в Redis"""
        await self.redis_client.setex(
            f"notification:{notification.id}",
            86400 * 30,  # 30 дней
            json.dumps(asdict(notification), default=str)
        )

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'smtp': {
            'host': 'smtp.gmail.com',
            'port': 587,
            'use_tls': True,
            'username': 'alerts@peper-binance.com',
            'password': 'your-password',
            'from': 'alerts@peper-binance.com'
        },
        'twilio_account_sid': 'your-twilio-sid',
        'twilio_auth_token': 'your-twilio-token',
        'twilio_phone_number': '+1234567890',
        'telegram_bot_token': 'your-telegram-bot-token',
        'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
        'discord_webhook_url': 'https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK'
    }
    
    notification_system = EnterpriseAlertNotificationSystem(config)
    await notification_system.start()
    
    print("Enterprise Alert & Notification System started")
    
    # Тестовый алерт
    test_alert = {
        'id': 'test_alert_001',
        'title': 'High CPU Usage',
        'description': 'CPU usage is above 90%',
        'severity': 'critical',
        'component': 'trading_engine',
        'metric_name': 'cpu_usage',
        'current_value': 95.5,
        'threshold_value': 90.0,
        'timestamp': datetime.now()
    }
    
    await notification_system.process_alert(test_alert)
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await notification_system.stop()

if __name__ == '__main__':
    asyncio.run(main())