from django.db import models

class TurkishTexts(models.Model):
    id = models.AutoField(primary_key=True)
    text = models.TextField()
    category = models.IntegerField()
    
    def __str__(self):
        return f'Text ID: {self.id}, Category: {self.category}'
    
class UploadedTurkishTexts(models.Model):
    text=models.TextField()
    category=models.IntegerField()
    
    def __str__(self):
        return f'Uploaded Text: {self.text}, Category: {self.category}'
    
class EnglishTexts(models.Model):
    id = models.AutoField(primary_key=True)
    text = models.TextField()
    category = models.IntegerField()

    def __str__(self):
        return f'Text ID: {self.id}, Category: {self.category}'

class UploadedEnglishTexts(models.Model):
    text=models.TextField()
    category=models.IntegerField()
    
    def __str__(self):
        return f'Uploaded Text: {self.text}, Category: {self.category}'