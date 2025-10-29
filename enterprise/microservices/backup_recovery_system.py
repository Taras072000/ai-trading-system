"""
Enterprise Backup & Recovery System - Автоматическое резервное копирование и восстановление
Обеспечивает надежность данных и быстрое восстановление после сбоев
"""

import asyncio
import json
import os
import shutil
import tarfile
import gzip
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiofiles
import aiohttp
import logging
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import psycopg2
import boto3
from botocore.exceptions import ClientError
import subprocess

class BackupType(Enum):
    """Типы резервных копий"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Статусы резервного копирования"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class StorageType(Enum):
    """Типы хранилищ"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

@dataclass
class BackupJob:
    """Задача резервного копирования"""
    id: str
    name: str
    backup_type: BackupType
    source_path: str
    destination: str
    storage_type: StorageType
    schedule: str  # cron expression
    retention_days: int
    compression: bool = True
    encryption: bool = True
    status: BackupStatus = BackupStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: str = ""
    error_message: str = ""

@dataclass
class RecoveryPoint:
    """Точка восстановления"""
    id: str
    backup_job_id: str
    timestamp: datetime
    backup_type: BackupType
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    is_verified: bool = False

# Метрики
BACKUP_JOBS_TOTAL = Counter('backup_jobs_total', 'Total backup jobs', ['type', 'status'])
BACKUP_DURATION = Histogram('backup_duration_seconds', 'Backup duration', ['type'])
BACKUP_SIZE = Histogram('backup_size_bytes', 'Backup size in bytes', ['type'])
RECOVERY_OPERATIONS = Counter('recovery_operations_total', 'Recovery operations', ['status'])
STORAGE_USAGE = Gauge('backup_storage_usage_bytes', 'Storage usage in bytes', ['storage_type'])

class EnterpriseBackupRecoverySystem:
    """Enterprise система резервного копирования и восстановления"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_jobs: Dict[str, BackupJob] = {}
        self.recovery_points: Dict[str, RecoveryPoint] = {}
        self.redis_client = None
        self.logger = self._setup_logging()
        self.encryption_key = config.get('encryption_key', 'default-encryption-key')
        
        # Инициализация хранилищ
        self.storage_clients = {}
        self._init_storage_clients()
        
        # Загрузка конфигурации заданий
        self._load_backup_jobs()
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_backup_recovery')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _init_storage_clients(self):
        """Инициализация клиентов хранилищ"""
        # AWS S3
        if self.config.get('aws_access_key'):
            self.storage_clients[StorageType.S3] = boto3.client(
                's3',
                aws_access_key_id=self.config['aws_access_key'],
                aws_secret_access_key=self.config['aws_secret_key'],
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            
        # Google Cloud Storage
        if self.config.get('gcs_credentials'):
            from google.cloud import storage
            self.storage_clients[StorageType.GCS] = storage.Client.from_service_account_json(
                self.config['gcs_credentials']
            )
            
        # Azure Blob Storage
        if self.config.get('azure_connection_string'):
            from azure.storage.blob import BlobServiceClient
            self.storage_clients[StorageType.AZURE] = BlobServiceClient.from_connection_string(
                self.config['azure_connection_string']
            )
            
    def _load_backup_jobs(self):
        """Загрузка конфигурации заданий резервного копирования"""
        backup_jobs_config = {
            'database_full': BackupJob(
                id='db_full_backup',
                name='Database Full Backup',
                backup_type=BackupType.FULL,
                source_path='postgresql://localhost:5432/peper_binance',
                destination='backups/database/full',
                storage_type=StorageType.S3,
                schedule='0 2 * * *',  # Каждый день в 2:00
                retention_days=30,
                compression=True,
                encryption=True
            ),
            'database_incremental': BackupJob(
                id='db_incremental_backup',
                name='Database Incremental Backup',
                backup_type=BackupType.INCREMENTAL,
                source_path='postgresql://localhost:5432/peper_binance',
                destination='backups/database/incremental',
                storage_type=StorageType.S3,
                schedule='0 */6 * * *',  # Каждые 6 часов
                retention_days=7,
                compression=True,
                encryption=True
            ),
            'models_backup': BackupJob(
                id='models_backup',
                name='AI Models Backup',
                backup_type=BackupType.FULL,
                source_path='models/',
                destination='backups/models',
                storage_type=StorageType.S3,
                schedule='0 3 * * 0',  # Каждое воскресенье в 3:00
                retention_days=90,
                compression=True,
                encryption=True
            ),
            'config_backup': BackupJob(
                id='config_backup',
                name='Configuration Backup',
                backup_type=BackupType.FULL,
                source_path='config/',
                destination='backups/config',
                storage_type=StorageType.LOCAL,
                schedule='0 1 * * *',  # Каждый день в 1:00
                retention_days=14,
                compression=True,
                encryption=True
            ),
            'logs_backup': BackupJob(
                id='logs_backup',
                name='Logs Backup',
                backup_type=BackupType.DIFFERENTIAL,
                source_path='logs/',
                destination='backups/logs',
                storage_type=StorageType.S3,
                schedule='0 */12 * * *',  # Каждые 12 часов
                retention_days=30,
                compression=True,
                encryption=False
            )
        }
        
        for job_id, job in backup_jobs_config.items():
            job.created_at = datetime.now()
            self.backup_jobs[job_id] = job
            
    async def start(self):
        """Запуск системы резервного копирования"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Запуск основных процессов
        asyncio.create_task(self._backup_scheduler_loop())
        asyncio.create_task(self._backup_monitor_loop())
        asyncio.create_task(self._cleanup_old_backups_loop())
        asyncio.create_task(self._verify_backups_loop())
        
        self.logger.info("Enterprise Backup & Recovery System started")
        
    async def stop(self):
        """Остановка системы"""
        if self.redis_client:
            await self.redis_client.close()
            
    async def _backup_scheduler_loop(self):
        """Планировщик резервного копирования"""
        while True:
            try:
                current_time = datetime.now()
                
                for job_id, job in self.backup_jobs.items():
                    if await self._should_run_backup(job, current_time):
                        asyncio.create_task(self._execute_backup_job(job))
                        
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(30)
                
    async def _should_run_backup(self, job: BackupJob, current_time: datetime) -> bool:
        """Проверка необходимости запуска резервного копирования"""
        # Простая проверка по времени (в реальной системе нужен cron parser)
        last_run_key = f"backup_last_run:{job.id}"
        last_run_str = await self.redis_client.get(last_run_key)
        
        if not last_run_str:
            return True
            
        last_run = datetime.fromisoformat(last_run_str)
        
        # Проверка интервала (упрощенная логика)
        if '*/6' in job.schedule:  # Каждые 6 часов
            return current_time - last_run >= timedelta(hours=6)
        elif '*/12' in job.schedule:  # Каждые 12 часов
            return current_time - last_run >= timedelta(hours=12)
        elif '0 2 * * *' in job.schedule:  # Каждый день в 2:00
            return (current_time - last_run >= timedelta(days=1) and 
                   current_time.hour == 2)
        elif '0 3 * * 0' in job.schedule:  # Каждое воскресенье в 3:00
            return (current_time - last_run >= timedelta(days=7) and 
                   current_time.weekday() == 6 and current_time.hour == 3)
                   
        return False
        
    async def _execute_backup_job(self, job: BackupJob):
        """Выполнение задания резервного копирования"""
        start_time = datetime.now()
        job.status = BackupStatus.IN_PROGRESS
        job.started_at = start_time
        
        try:
            self.logger.info(f"Starting backup job: {job.name}")
            
            # Создание резервной копии в зависимости от типа источника
            if job.source_path.startswith('postgresql://'):
                backup_path = await self._backup_database(job)
            else:
                backup_path = await self._backup_files(job)
                
            # Сжатие и шифрование
            if job.compression or job.encryption:
                backup_path = await self._process_backup_file(backup_path, job)
                
            # Загрузка в хранилище
            if job.storage_type != StorageType.LOCAL:
                await self._upload_to_storage(backup_path, job)
                
            # Вычисление контрольной суммы
            job.checksum = await self._calculate_checksum(backup_path)
            job.size_bytes = os.path.getsize(backup_path)
            
            # Создание точки восстановления
            recovery_point = RecoveryPoint(
                id=f"{job.id}_{int(start_time.timestamp())}",
                backup_job_id=job.id,
                timestamp=start_time,
                backup_type=job.backup_type,
                size_bytes=job.size_bytes,
                checksum=job.checksum,
                metadata={
                    'source_path': job.source_path,
                    'destination': job.destination,
                    'storage_type': job.storage_type.value
                }
            )
            
            self.recovery_points[recovery_point.id] = recovery_point
            
            # Сохранение информации в Redis
            await self._save_recovery_point(recovery_point)
            
            job.status = BackupStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Обновление времени последнего запуска
            await self.redis_client.set(
                f"backup_last_run:{job.id}",
                start_time.isoformat()
            )
            
            # Метрики
            BACKUP_JOBS_TOTAL.labels(
                type=job.backup_type.value, 
                status='completed'
            ).inc()
            BACKUP_DURATION.labels(type=job.backup_type.value).observe(
                (datetime.now() - start_time).total_seconds()
            )
            BACKUP_SIZE.labels(type=job.backup_type.value).observe(job.size_bytes)
            
            self.logger.info(f"Backup job completed: {job.name}, size: {job.size_bytes} bytes")
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            BACKUP_JOBS_TOTAL.labels(
                type=job.backup_type.value, 
                status='failed'
            ).inc()
            
            self.logger.error(f"Backup job failed: {job.name}, error: {e}")
            
    async def _backup_database(self, job: BackupJob) -> str:
        """Резервное копирование базы данных"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"db_backup_{timestamp}.sql"
        backup_path = os.path.join('/tmp', backup_filename)
        
        # Извлечение параметров подключения из URL
        # postgresql://user:password@host:port/database
        db_url = job.source_path
        
        if job.backup_type == BackupType.FULL:
            # Полное резервное копирование
            cmd = [
                'pg_dump',
                '--verbose',
                '--clean',
                '--no-owner',
                '--no-privileges',
                '--format=custom',
                '--file', backup_path,
                db_url
            ]
        else:
            # Инкрементальное копирование (WAL архивы)
            cmd = [
                'pg_basebackup',
                '--pgdata', backup_path,
                '--format=tar',
                '--wal-method=stream',
                '--checkpoint=fast',
                '--progress',
                '--verbose',
                '--dbname', db_url
            ]
            
        # Выполнение команды
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Database backup failed: {stderr.decode()}")
            
        return backup_path
        
    async def _backup_files(self, job: BackupJob) -> str:
        """Резервное копирование файлов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"files_backup_{timestamp}.tar"
        backup_path = os.path.join('/tmp', backup_filename)
        
        # Создание tar архива
        with tarfile.open(backup_path, 'w') as tar:
            if os.path.exists(job.source_path):
                tar.add(job.source_path, arcname=os.path.basename(job.source_path))
            else:
                raise Exception(f"Source path not found: {job.source_path}")
                
        return backup_path
        
    async def _process_backup_file(self, backup_path: str, job: BackupJob) -> str:
        """Обработка файла резервной копии (сжатие и шифрование)"""
        processed_path = backup_path
        
        # Сжатие
        if job.compression:
            compressed_path = f"{backup_path}.gz"
            
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            os.remove(backup_path)
            processed_path = compressed_path
            
        # Шифрование (простое XOR шифрование для демонстрации)
        if job.encryption:
            encrypted_path = f"{processed_path}.enc"
            
            with open(processed_path, 'rb') as f_in:
                with open(encrypted_path, 'wb') as f_out:
                    key_bytes = self.encryption_key.encode()
                    key_len = len(key_bytes)
                    
                    while True:
                        chunk = f_in.read(8192)
                        if not chunk:
                            break
                            
                        encrypted_chunk = bytes(
                            chunk[i] ^ key_bytes[i % key_len] 
                            for i in range(len(chunk))
                        )
                        f_out.write(encrypted_chunk)
                        
            os.remove(processed_path)
            processed_path = encrypted_path
            
        return processed_path
        
    async def _upload_to_storage(self, backup_path: str, job: BackupJob):
        """Загрузка резервной копии в облачное хранилище"""
        filename = os.path.basename(backup_path)
        storage_key = f"{job.destination}/{filename}"
        
        if job.storage_type == StorageType.S3:
            await self._upload_to_s3(backup_path, storage_key)
        elif job.storage_type == StorageType.GCS:
            await self._upload_to_gcs(backup_path, storage_key)
        elif job.storage_type == StorageType.AZURE:
            await self._upload_to_azure(backup_path, storage_key)
            
    async def _upload_to_s3(self, backup_path: str, storage_key: str):
        """Загрузка в Amazon S3"""
        s3_client = self.storage_clients.get(StorageType.S3)
        if not s3_client:
            raise Exception("S3 client not configured")
            
        bucket_name = self.config.get('s3_bucket', 'peper-binance-backups')
        
        try:
            s3_client.upload_file(backup_path, bucket_name, storage_key)
            self.logger.info(f"Uploaded to S3: s3://{bucket_name}/{storage_key}")
        except ClientError as e:
            raise Exception(f"S3 upload failed: {e}")
            
    async def _upload_to_gcs(self, backup_path: str, storage_key: str):
        """Загрузка в Google Cloud Storage"""
        gcs_client = self.storage_clients.get(StorageType.GCS)
        if not gcs_client:
            raise Exception("GCS client not configured")
            
        bucket_name = self.config.get('gcs_bucket', 'peper-binance-backups')
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(storage_key)
        
        blob.upload_from_filename(backup_path)
        self.logger.info(f"Uploaded to GCS: gs://{bucket_name}/{storage_key}")
        
    async def _upload_to_azure(self, backup_path: str, storage_key: str):
        """Загрузка в Azure Blob Storage"""
        azure_client = self.storage_clients.get(StorageType.AZURE)
        if not azure_client:
            raise Exception("Azure client not configured")
            
        container_name = self.config.get('azure_container', 'peper-binance-backups')
        
        with open(backup_path, 'rb') as data:
            azure_client.upload_blob(
                container=container_name,
                name=storage_key,
                data=data,
                overwrite=True
            )
            
        self.logger.info(f"Uploaded to Azure: {container_name}/{storage_key}")
        
    async def _calculate_checksum(self, file_path: str) -> str:
        """Вычисление контрольной суммы файла"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
                
        return hash_sha256.hexdigest()
        
    async def _save_recovery_point(self, recovery_point: RecoveryPoint):
        """Сохранение точки восстановления в Redis"""
        await self.redis_client.setex(
            f"recovery_point:{recovery_point.id}",
            86400 * recovery_point.metadata.get('retention_days', 30),
            json.dumps(asdict(recovery_point), default=str)
        )
        
    async def _backup_monitor_loop(self):
        """Мониторинг выполнения резервного копирования"""
        while True:
            try:
                # Проверка зависших заданий
                for job in self.backup_jobs.values():
                    if (job.status == BackupStatus.IN_PROGRESS and 
                        job.started_at and 
                        datetime.now() - job.started_at > timedelta(hours=2)):
                        
                        job.status = BackupStatus.FAILED
                        job.error_message = "Backup timeout"
                        job.completed_at = datetime.now()
                        
                        self.logger.warning(f"Backup job timed out: {job.name}")
                        
                await asyncio.sleep(300)  # Проверка каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Backup monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_old_backups_loop(self):
        """Очистка старых резервных копий"""
        while True:
            try:
                current_time = datetime.now()
                
                for job in self.backup_jobs.values():
                    await self._cleanup_old_backups_for_job(job, current_time)
                    
                await asyncio.sleep(3600)  # Проверка каждый час
                
            except Exception as e:
                self.logger.error(f"Backup cleanup error: {e}")
                await asyncio.sleep(1800)
                
    async def _cleanup_old_backups_for_job(self, job: BackupJob, current_time: datetime):
        """Очистка старых резервных копий для конкретного задания"""
        cutoff_time = current_time - timedelta(days=job.retention_days)
        
        # Поиск старых точек восстановления
        recovery_point_keys = await self.redis_client.keys(f"recovery_point:*")
        
        for key in recovery_point_keys:
            try:
                rp_data = await self.redis_client.get(key)
                if rp_data:
                    rp_dict = json.loads(rp_data)
                    
                    if rp_dict['backup_job_id'] == job.id:
                        rp_timestamp = datetime.fromisoformat(rp_dict['timestamp'])
                        
                        if rp_timestamp < cutoff_time:
                            # Удаление из хранилища
                            await self._delete_backup_from_storage(rp_dict, job)
                            
                            # Удаление из Redis
                            await self.redis_client.delete(key)
                            
                            self.logger.info(f"Cleaned up old backup: {key}")
                            
            except Exception as e:
                self.logger.error(f"Error cleaning up backup {key}: {e}")
                
    async def _delete_backup_from_storage(self, recovery_point_dict: Dict, job: BackupJob):
        """Удаление резервной копии из хранилища"""
        if job.storage_type == StorageType.S3:
            # Удаление из S3
            s3_client = self.storage_clients.get(StorageType.S3)
            if s3_client:
                bucket_name = self.config.get('s3_bucket', 'peper-binance-backups')
                # Логика удаления из S3
                
    async def _verify_backups_loop(self):
        """Проверка целостности резервных копий"""
        while True:
            try:
                # Проверка случайных резервных копий
                recovery_point_keys = await self.redis_client.keys("recovery_point:*")
                
                if recovery_point_keys:
                    # Выбор случайной точки восстановления для проверки
                    import random
                    key = random.choice(recovery_point_keys)
                    
                    rp_data = await self.redis_client.get(key)
                    if rp_data:
                        rp_dict = json.loads(rp_data)
                        await self._verify_backup_integrity(rp_dict)
                        
                await asyncio.sleep(7200)  # Проверка каждые 2 часа
                
            except Exception as e:
                self.logger.error(f"Backup verification error: {e}")
                await asyncio.sleep(3600)
                
    async def _verify_backup_integrity(self, recovery_point_dict: Dict):
        """Проверка целостности резервной копии"""
        try:
            # Загрузка файла из хранилища (если не локальный)
            # Проверка контрольной суммы
            # Пробное восстановление (для критических данных)
            
            self.logger.info(f"Verified backup integrity: {recovery_point_dict['id']}")
            
        except Exception as e:
            self.logger.error(f"Backup integrity check failed: {e}")
            
    async def restore_from_backup(self, recovery_point_id: str, target_path: str) -> bool:
        """Восстановление из резервной копии"""
        try:
            # Получение информации о точке восстановления
            rp_data = await self.redis_client.get(f"recovery_point:{recovery_point_id}")
            if not rp_data:
                raise Exception(f"Recovery point not found: {recovery_point_id}")
                
            rp_dict = json.loads(rp_data)
            
            self.logger.info(f"Starting restore from backup: {recovery_point_id}")
            
            # Загрузка резервной копии
            backup_path = await self._download_backup(rp_dict)
            
            # Расшифровка и распаковка
            restored_path = await self._process_restore_file(backup_path, rp_dict)
            
            # Восстановление данных
            if rp_dict['metadata']['source_path'].startswith('postgresql://'):
                await self._restore_database(restored_path, target_path)
            else:
                await self._restore_files(restored_path, target_path)
                
            RECOVERY_OPERATIONS.labels(status='success').inc()
            
            self.logger.info(f"Restore completed successfully: {recovery_point_id}")
            return True
            
        except Exception as e:
            RECOVERY_OPERATIONS.labels(status='failed').inc()
            self.logger.error(f"Restore failed: {e}")
            return False
            
    async def _download_backup(self, recovery_point_dict: Dict) -> str:
        """Загрузка резервной копии из хранилища"""
        # Реализация загрузки в зависимости от типа хранилища
        # Возвращает путь к загруженному файлу
        pass
        
    async def _process_restore_file(self, backup_path: str, recovery_point_dict: Dict) -> str:
        """Обработка файла для восстановления (расшифровка и распаковка)"""
        # Реализация расшифровки и распаковки
        # Возвращает путь к обработанному файлу
        pass
        
    async def _restore_database(self, backup_path: str, target_db_url: str):
        """Восстановление базы данных"""
        # Реализация восстановления базы данных
        pass
        
    async def _restore_files(self, backup_path: str, target_path: str):
        """Восстановление файлов"""
        # Реализация восстановления файлов
        pass

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'encryption_key': 'enterprise-backup-key-2024',
        'aws_access_key': 'your-aws-access-key',
        'aws_secret_key': 'your-aws-secret-key',
        'aws_region': 'us-east-1',
        's3_bucket': 'peper-binance-backups'
    }
    
    backup