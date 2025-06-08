import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, throwError } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  class_name: string;
  conf: number;
}

export interface PredictionResponse {
  image: string; // base64
  boxes: Box[];
}

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private apiUrl = 'http://127.0.0.1:8000';

  private loading = new BehaviorSubject<boolean>(false);
  loading$ = this.loading.asObservable();

  constructor(private http: HttpClient) { }

  getModels(): Observable<string[]> {
    return this.http.get<string[]>(`${this.apiUrl}/models`).pipe(
      catchError(this.handleError)
    );
  }

  detect(model: string, image: File): Observable<PredictionResponse> {
    const formData = new FormData();
    formData.append('model', model);
    formData.append('image', image);

    this.loading.next(true);

    return this.http.post<PredictionResponse>(`${this.apiUrl}/detect`, formData).pipe(
      tap(() => this.loading.next(false)),
      catchError(err => {
        this.loading.next(false);
        return this.handleError(err);
      })
    );
  }

  private handleError(error: any) {
    console.error('API Error:', error);
    return throwError(() => new Error('Something bad happened; please try again later.'));
  }
}
