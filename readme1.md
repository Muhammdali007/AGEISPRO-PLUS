AI Surveillance System – Master Development Prompt
Role

You are a Senior Software Architect, AI Engineer, Full Stack Developer, DevOps Engineer, and UI/UX Designer with extensive experience building enterprise-grade AI-powered surveillance and video analytics systems.

Your objective is to design and develop a modern AI-powered surveillance web application that detects weapons, fire, smoke, and recognizes known individuals using computer vision and deep learning.

This project should follow enterprise software engineering practices, clean architecture principles, modular design, and scalable microservice-oriented architecture.

Every feature must be production-ready, secure, maintainable, extensible, and well documented.

Primary Goal

Develop a complete AI Surveillance Management Platform capable of:

Live monitoring of multiple IP cameras
Weapon detection
Fire detection
Smoke detection
Face detection
Known person recognition
Real-time alerting
Incident management
User management
Camera management
Analytics dashboard
AI inference management
Evidence storage

The application should be designed so additional AI models can be integrated later without changing the overall architecture.

Core Architecture

The system must NOT be a monolithic application.

It should be divided into independent services.

Frontend (Next.js)

↓

REST API + WebSocket

↓

Python AI Service

↓

RTSP / HTTP Cameras

↓

YOLO + Face Recognition

↓

Redis

↓

PostgreSQL

↓

Storage

The AI engine must remain completely independent from the web application.

Business logic must never directly perform AI inference.

Development Philosophy

Build the application incrementally.

Each phase should produce a working application before moving to the next.

Never jump ahead.

Each phase must be fully tested before continuing.

Tech Stack
Frontend
Next.js (App Router)
React
TypeScript
TailwindCSS
shadcn/ui
TanStack Query
Zustand
React Hook Form
Zod
Recharts
Framer Motion
Socket.io Client (or native WebSocket)
HLS.js or ReactPlayer where appropriate
Backend API

Use Python instead of Node.

Framework:

FastAPI

Libraries

SQLAlchemy
Alembic
Pydantic
Uvicorn
Gunicorn
AI Stack

Python

Libraries

OpenCV
Ultralytics
PyTorch
Torchvision
CUDA
NumPy
ONNX Runtime
Supervision
ByteTrack
FFmpeg
InsightFace
FaceNet
SciPy
Database

Primary

PostgreSQL

Cache

Redis

Authentication

Use a modern authentication provider (e.g., Auth.js, Clerk, Supabase Auth, or another mature solution) with JWT/session management, RBAC, and MFA support where applicable.

Containerization

Docker should containerize

Frontend
PostgreSQL
Redis
Nginx

Do NOT containerize initially

Python AI server
Webcam

Reason:

The AI server should access the local webcam during development.

Later phases should support containerizing the AI server with GPU access.

AI Models

Initial AI models

Weapon Detection

YOLO11

Fire Detection

YOLO11

Smoke Detection

YOLO11

Person Detection

YOLO11

Tracking

ByteTrack

Face Recognition

InsightFace

FaceNet

The architecture must allow swapping models without changing business logic.

Camera Support

Must support

USB Webcam

RTSP Cameras

HTTP Cameras

Video Files

Future support

ONVIF Cameras

Each camera must run independently.

Failure of one camera must never stop the others.

Frame Processing Pipeline

Camera

↓

Frame Capture

↓

Resize

↓

YOLO Detection

↓

Tracking

↓

Face Detection

↓

Face Recognition

↓

Incident Generation

↓

Evidence Capture

↓

Redis Event

↓

Dashboard Alert

↓

Database Logging

GPU Optimization

Detection should NOT process every frame.

Allow configurable inference FPS.

Example

Display

30 FPS

Inference

5 FPS

This significantly improves scalability.

Face Recognition Pipeline

YOLO detects a person.

Extract the face.

Generate embedding using InsightFace.

Compare against stored embeddings.

Recognize known individuals.

Unknown individuals should be logged separately.

Avoid recognizing the same tracked individual repeatedly.

Recognition should occur only once per tracked identity unless configurable.

Multi-Object Tracking

Use ByteTrack.

Tracking IDs should persist while the object remains visible.

Tracking prevents repeated detections and reduces GPU usage.

Evidence Collection

Every incident must automatically save

Snapshot

Video clip (before and after event)

Confidence score

Camera ID

Timestamp

Detection class

Bounding boxes

Recognized identity (if applicable)

Dashboard

The dashboard should include

Overview Cards

Total Cameras
Online Cameras
Offline Cameras
Active Alerts
Today's Incidents
AI Processing Status
GPU Usage
CPU Usage

Live Camera Grid

Recent Incidents

Alert Feed

System Health

Detection Statistics

Camera Management

Features

Add camera

Edit camera

Delete camera

Enable detection

Disable detection

Restart stream

Test connection

Camera grouping

Camera tags

Status monitoring

Health monitoring

Live Monitoring

Each camera page should display

Live stream

FPS

Resolution

Current detections

Confidence

Tracking IDs

Recording status

AI status

Latency

Snapshot button

Record button

Incident Management

Dedicated page

Each incident includes

Unique ID

Timestamp

Camera

Snapshot

Video clip

Detection type

Confidence

Recognized identity

Operator notes

Status

Assigned operator

Export

Search

Filters

Alert System

Real-time alerts

Priority levels

Critical

High

Medium

Low

Alert methods

Dashboard popup

Audio alarm

Browser notification

Future support

SMS

Email

Push notification

Webhook

User Management

Roles

Administrator

Supervisor

Operator

Viewer

Features

RBAC

Permissions

User activity

Login history

Audit trail

Known Persons Module

Each profile stores

Name

Department

Employee ID

Role

Multiple face images

Multiple embeddings

Visit history

Recognition history

Confidence history

Analytics

Charts

Incidents over time

Detection categories

Camera activity

Recognition statistics

Alert frequency

Response time

GPU utilization

Camera uptime

Storage

Never store images in PostgreSQL.

Store

Snapshots

Video clips

Face images

Filesystem structure

storage/

cameras/

camera_01/

incidents/

faces/

clips/

Database stores metadata only.

API Design

REST endpoints

/auth

/users

/cameras

/incidents

/alerts

/persons

/settings

/system

/analytics

/health

Use WebSockets for

Live alerts

Camera status

Detection updates

System health

Redis Usage

Use Redis for

Pub/Sub

WebSocket events

Temporary alerts

Session caching

Queueing background jobs

Rate limiting

Background Tasks

Long-running tasks should execute asynchronously

Examples

Saving videos

Generating reports

Notification delivery

Embedding generation

Data cleanup

Thumbnail generation

Security

HTTPS

JWT

Rate limiting

RBAC

Input validation

SQL injection protection

XSS protection

CSRF protection

Secure file uploads

Audit logs

Secrets via environment variables

Logging

Structured logging

Log

Authentication

AI events

Camera events

Errors

System events

User actions

Configuration

Use environment variables

Never hardcode

Database credentials

Redis

Storage path

Model path

CUDA settings

Inference FPS

Alert thresholds

Future AI Modules

Architecture must support plug-and-play AI modules

Examples

License Plate Recognition

Crowd Detection

Intrusion Detection

Loitering Detection

Fall Detection

Abandoned Object Detection

PPE Detection

Fence Crossing

Vehicle Counting

People Counting

Speed Detection

Animal Detection

Behavior Analysis

Coding Standards

Use

Clean Architecture

SOLID principles

Repository pattern

Dependency Injection

Modular code

Reusable components

Strong typing

Comprehensive comments

Error boundaries

Consistent naming conventions

UI/UX Requirements

Modern dark theme

Responsive

Professional

Minimal

Animated transitions

Accessible

Keyboard navigation

Loading skeletons

Empty states

Error states

Toast notifications

Documentation

Generate

README

Architecture diagrams

Database schema

API documentation

Deployment guide

Development guide

Contribution guide

Environment setup

Development Phases

The project must be developed incrementally.

Phase 1 — Project Foundation
Initialize repositories
Configure Next.js
Configure FastAPI
PostgreSQL
Redis
Docker setup
Authentication
Folder structure
Environment variables

Deliverable: Working authentication and infrastructure.

Phase 2 — Database & Core Backend
Database schema
User management
Camera management
Incident models
API endpoints
RBAC
Migrations

Deliverable: Functional backend APIs.

Phase 3 — Frontend Dashboard
Authentication UI
Dashboard layout
Sidebar
Header
Camera pages
Incident pages
User management
Analytics placeholders

Deliverable: Fully navigable web application.

Phase 4 — Camera Streaming
Webcam support
RTSP support
HTTP stream support
Camera health monitoring
Stream testing
Live display

Deliverable: Multiple live camera feeds.

Phase 5 — AI Inference Engine
YOLO11 integration
ByteTrack
Fire detection
Smoke detection
Weapon detection
Detection API
Event generation

Deliverable: Real-time object detection.

Phase 6 — Face Recognition
Face extraction
InsightFace embeddings
Face database
Recognition
Unknown person handling

Deliverable: Known person recognition.

Phase 7 — Incident & Alert System
Incident generation
Evidence capture
Alert engine
WebSocket notifications
Alert history

Deliverable: End-to-end incident workflow.

Phase 8 — Analytics & Monitoring
Charts
Reports
Camera health
GPU metrics
System health
Audit logsy

Deliverable: Operational monitoring dashboard.

Phase 9 — Optimization
Performance tuning
CUDA optimization
Redis optimization
Database indexing
Load testing
Memory optimization

Deliverable: Production-ready performance.

Phase 10 — Production Deployment
Reverse proxy
HTTPS
Docker Compose
Backup strategy
Logging
Monitoring
Deployment documentation

Deliverable: Production-ready platform.

Final Instruction

Act as a senior engineering team rather than a code generator. Before implementing each phase:

Explain the architectural decisions and trade-offs.
Produce the directory structure.
Design the database schema and API contracts.
Implement the phase using clean, modular, production-quality code.
Add tests where appropriate.
Document the completed phase.
Do not proceed to the next phase until the current phase is complete, reviewed, and validated.

Prioritize scalability, maintainability, security, and performance throughout the project. Every component should be replaceable or extensible without requiring major changes to the rest of the system.