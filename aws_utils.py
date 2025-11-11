"""Simple AWS utilities for S3 and DynamoDB"""
import boto3
import streamlit as st
from botocore.exceptions import ClientError
import time

class S3Manager:
    """Manage PDF storage in S3"""
    
    def __init__(self, bucket_name, region='us-east-1'):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=region
        )
    
    def upload_pdf(self, file_bytes, filename):
        """Upload PDF to S3"""
        try:
            key = f"pdfs/{filename}"
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_bytes,
                ContentType='application/pdf'
            )
            return key
        except ClientError as e:
            st.error(f"Upload failed: {e}")
            return None
    
    def list_pdfs(self):
        """List all PDFs"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='pdfs/'
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            st.error(f"List failed: {e}")
            return []


class DynamoDBChat:
    """Manage chat history in DynamoDB"""
    
    def __init__(self, table_name='ChatHistory', region='us-east-1'):
        self.dynamodb = boto3.resource('dynamodb',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=region
        )
        self.table = self.dynamodb.Table(table_name)
    
    def save_message(self, session_id, role, content):
        """Save a message"""
        try:
            self.table.put_item(Item={
                'session_id': session_id,
                'timestamp': int(time.time() * 1000),
                'role': role,
                'content': content,
                'ttl': int(time.time()) + (30 * 86400)
            })
            return True
        except ClientError as e:
            st.error(f"Save failed: {e}")
            return False
    
    def get_history(self, session_id, limit=20):
        """Get chat history"""
        try:
            response = self.table.query(
                KeyConditionExpression='session_id = :sid',
                ExpressionAttributeValues={':sid': session_id},
                ScanIndexForward=False,
                Limit=limit
            )
            messages = response.get('Items', [])
            return sorted(messages, key=lambda x: x['timestamp'])
        except ClientError as e:
            st.error(f"Load failed: {e}")
            return []
    
    def clear_history(self, session_id):
        """Clear chat history"""
        try:
            messages = self.get_history(session_id, limit=100)
            with self.table.batch_writer() as batch:
                for msg in messages:
                    batch.delete_item(Key={
                        'session_id': msg['session_id'],
                        'timestamp': msg['timestamp']
                    })
            return True
        except ClientError as e:
            st.error(f"Clear failed: {e}")
            return False
