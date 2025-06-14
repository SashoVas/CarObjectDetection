import { Component, OnInit, ElementRef, ViewChild, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictionService, PredictionResponse, Box, VideoProcessResponse } from './services/prediction';
import { Observable } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  @ViewChild('originalImage') originalImageRef!: ElementRef<HTMLImageElement>;
  @ViewChild('carousel') carouselRef!: ElementRef<HTMLCanvasElement>;

  models$: Observable<string[]>;
  loading$: Observable<boolean>;

  selectedModel: string = '';
  selectedFile: File | null = null;
  selectedFileType: 'image' | 'video' | null = null;
  originalImageSrc: string | ArrayBuffer | null = null;
  originalVideoSrc: string | null = null;
  predictionResult: PredictionResponse | null = null;
  processedVideoUrl: SafeUrl | null = null;
  errorMessage: string | null = null;
  croppedDetections: { imageData: string, label: string }[] = [];

  constructor(private predictionService: PredictionService, private cdr: ChangeDetectorRef, private sanitizer: DomSanitizer) {
    this.models$ = this.predictionService.getModels();
    this.loading$ = this.predictionService.loading$;
  }

  ngOnInit() {
    this.models$.subscribe(models => {
      if (models && models.length > 0) {
        this.selectedModel = models[0];
      }
    });
  }

  onFileSelected(event: Event) {
    const element = event.currentTarget as HTMLInputElement;
    let fileList: FileList | null = element.files;
    if (fileList && fileList.length > 0) {
      this.selectedFile = fileList[0];
      this.selectedFileType = this.selectedFile.type.startsWith('image/') ? 'image' : 'video';
      
      // Reset states
      this.predictionResult = null;
      this.errorMessage = null;
      this.croppedDetections = [];
      this.originalImageSrc = null;
      this.originalVideoSrc = null;
      this.processedVideoUrl = null;

      if (this.selectedFileType === 'image') {
        const reader = new FileReader();
        reader.onload = e => this.originalImageSrc = reader.result;
        reader.readAsDataURL(this.selectedFile);
      } else if (this.selectedFileType === 'video') {
        this.originalVideoSrc = URL.createObjectURL(this.selectedFile);
      }
    }
  }

  onPredict() {
    if (!this.selectedFile || !this.selectedModel) {
      this.errorMessage = "Please select a model and a file.";
      return;
    }

    this.errorMessage = null;
    this.predictionResult = null;
    this.processedVideoUrl = null;
    this.croppedDetections = [];

    if (this.selectedFileType === 'image') {
      this.predictionService.detect(this.selectedModel, this.selectedFile).subscribe({
        next: (result) => {
          this.predictionResult = result;
          this.generateCroppedImages();
        },
        error: (err) => {
          this.errorMessage = err.message;
        }
      });
    } else if (this.selectedFileType === 'video') {
      this.predictionService.processVideo(this.selectedModel, this.selectedFile).subscribe({
        next: (response: VideoProcessResponse) => {

          this.processedVideoUrl ='http://localhost:8000/' + response.result_path;
          this.predictionService.setLoading(false);
        },
        error: (err) => {
          this.errorMessage = err.message;
        }
      });
    }
  }

  generateCroppedImages() {
    if (!this.predictionResult || !this.originalImageRef) {
      return;
    }

    const imageElement = this.originalImageRef.nativeElement;
    const originalImage = new Image();
    originalImage.onload = () => {
      this.croppedDetections = this.predictionResult!.boxes.map(box => {
        const canvas = document.createElement('canvas');
        canvas.width = box.w;
        canvas.height = box.h;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(originalImage, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
          return {
            imageData: canvas.toDataURL(),
            label: box.class_name
          };
        }
        return { imageData: '', label: '' };
      });
      this.croppedDetections = this.croppedDetections.filter(d => d.imageData);
      
      this.cdr.detectChanges(); // Manually trigger change detection
    };
    originalImage.src = this.originalImageSrc as string;
  }
}
