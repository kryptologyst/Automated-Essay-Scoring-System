// JavaScript for Essay Scoring System

class EssayScoringApp {
    constructor() {
        this.currentSection = 'scoring';
        this.essays = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadEssays();
        this.loadAnalytics();
    }

    setupEventListeners() {
        // Scoring form
        document.getElementById('scoring-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.scoreEssay();
        });

        // Search functionality
        document.getElementById('search-essays').addEventListener('input', (e) => {
            this.searchEssays(e.target.value);
        });

        // Add essay form
        document.getElementById('add-essay-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addEssay();
        });
    }

    // Navigation
    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.style.display = 'none';
        });

        // Show selected section
        document.getElementById(`${sectionName}-section`).style.display = 'block';
        this.currentSection = sectionName;

        // Update active nav item
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        event.target.classList.add('active');

        // Load section-specific data
        if (sectionName === 'essays') {
            this.loadEssays();
        } else if (sectionName === 'analytics') {
            this.loadAnalytics();
        }
    }

    // Essay Scoring
    async scoreEssay() {
        const title = document.getElementById('essay-title').value;
        const content = document.getElementById('essay-content').value;

        if (!content.trim()) {
            this.showAlert('Please enter essay content', 'warning');
            return;
        }

        if (content.length < 10) {
            this.showAlert('Essay content must be at least 10 characters', 'warning');
            return;
        }

        try {
            this.showLoading('score-results', 'Scoring essay...');

            const response = await fetch('/api/score', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: content,
                    title: title || 'Untitled Essay'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayScoreResults(result);

        } catch (error) {
            console.error('Error scoring essay:', error);
            this.showAlert('Failed to score essay. Please try again.', 'danger');
            this.hideLoading('score-results');
        }
    }

    displayScoreResults(result) {
        const container = document.getElementById('score-results');
        
        // Determine score category and color
        let scoreClass, scoreLabel;
        if (result.score >= 8.5) {
            scoreClass = 'score-excellent';
            scoreLabel = 'Excellent';
        } else if (result.score >= 7.0) {
            scoreClass = 'score-good';
            scoreLabel = 'Good';
        } else if (result.score >= 5.5) {
            scoreClass = 'score-average';
            scoreLabel = 'Average';
        } else {
            scoreClass = 'score-poor';
            scoreLabel = 'Needs Improvement';
        }

        container.innerHTML = `
            <div class="score-display fade-in">
                <div class="score-circle ${scoreClass}">
                    ${result.score.toFixed(1)}
                </div>
                <h4>${scoreLabel}</h4>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
                <small class="text-muted">Confidence: ${(result.confidence * 100).toFixed(0)}%</small>
            </div>
            
            <div class="mt-4">
                <h6>Analysis</h6>
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-value">${result.analysis.word_count}</div>
                        <div class="feature-label">Words</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-value">${result.analysis.readability.toFixed(0)}</div>
                        <div class="feature-label">Readability</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-value">${(result.analysis.vocabulary_diversity * 100).toFixed(1)}%</div>
                        <div class="feature-label">Vocabulary Diversity</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-value">${result.analysis.avg_word_length.toFixed(1)}</div>
                        <div class="feature-label">Avg Word Length</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-value">${result.analysis.sentence_count}</div>
                        <div class="feature-label">Sentences</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-value">${result.analysis.grammar_score.toFixed(0)}</div>
                        <div class="feature-label">Grammar Score</div>
                    </div>
                </div>
            </div>
        `;
    }

    // Essay Management
    async loadEssays() {
        try {
            this.showLoading('essays-list', 'Loading essays...');

            const response = await fetch('/api/essays');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.essays = await response.json();
            this.displayEssays(this.essays);

        } catch (error) {
            console.error('Error loading essays:', error);
            this.showAlert('Failed to load essays', 'danger');
            this.hideLoading('essays-list');
        }
    }

    displayEssays(essays) {
        const container = document.getElementById('essays-list');
        
        if (essays.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-file-alt fa-3x mb-3"></i>
                    <h5>No essays found</h5>
                    <p>Add some essays to get started</p>
                </div>
            `;
            return;
        }

        container.innerHTML = essays.map(essay => `
            <div class="essay-item fade-in">
                <div class="essay-title">${this.escapeHtml(essay.title)}</div>
                <div class="essay-preview">${this.escapeHtml(essay.content.substring(0, 150))}...</div>
                <div class="essay-meta">
                    <div>
                        <span class="badge bg-secondary me-2">${essay.grade_level || 'N/A'}</span>
                        <span class="text-muted">${essay.word_count} words</span>
                    </div>
                    <div>
                        <span class="essay-score ${this.getScoreClass(essay.score)}">${essay.score.toFixed(1)}</span>
                        <button class="btn btn-sm btn-outline-danger ms-2" onclick="app.deleteEssay(${essay.id})">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    searchEssays(query) {
        if (!query.trim()) {
            this.displayEssays(this.essays);
            return;
        }

        const filteredEssays = this.essays.filter(essay => 
            essay.title.toLowerCase().includes(query.toLowerCase()) ||
            essay.content.toLowerCase().includes(query.toLowerCase())
        );

        this.displayEssays(filteredEssays);
    }

    showAddEssayForm() {
        const modal = new bootstrap.Modal(document.getElementById('addEssayModal'));
        modal.show();
    }

    async addEssay() {
        const title = document.getElementById('new-essay-title').value;
        const content = document.getElementById('new-essay-content').value;
        const prompt = document.getElementById('new-essay-prompt').value;
        const gradeLevel = document.getElementById('new-essay-grade').value;
        const score = parseFloat(document.getElementById('new-essay-score').value);

        if (!title.trim() || !content.trim()) {
            this.showAlert('Please fill in all required fields', 'warning');
            return;
        }

        try {
            const response = await fetch('/api/essays', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: title,
                    content: content,
                    prompt: prompt || null,
                    grade_level: gradeLevel || null,
                    score: score
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const newEssay = await response.json();
            this.essays.unshift(newEssay);
            this.displayEssays(this.essays);

            // Close modal and reset form
            const modal = bootstrap.Modal.getInstance(document.getElementById('addEssayModal'));
            modal.hide();
            document.getElementById('add-essay-form').reset();

            this.showAlert('Essay added successfully!', 'success');

        } catch (error) {
            console.error('Error adding essay:', error);
            this.showAlert('Failed to add essay', 'danger');
        }
    }

    async deleteEssay(essayId) {
        if (!confirm('Are you sure you want to delete this essay?')) {
            return;
        }

        try {
            const response = await fetch(`/api/essays/${essayId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.essays = this.essays.filter(essay => essay.id !== essayId);
            this.displayEssays(this.essays);
            this.showAlert('Essay deleted successfully', 'success');

        } catch (error) {
            console.error('Error deleting essay:', error);
            this.showAlert('Failed to delete essay', 'danger');
        }
    }

    // Analytics
    async loadAnalytics() {
        try {
            this.showLoading('analytics-content', 'Loading analytics...');

            const response = await fetch('/api/analytics');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const analytics = await response.json();
            this.displayAnalytics(analytics);

        } catch (error) {
            console.error('Error loading analytics:', error);
            this.showAlert('Failed to load analytics', 'danger');
            this.hideLoading('analytics-content');
        }
    }

    displayAnalytics(analytics) {
        const container = document.getElementById('analytics-content');
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${analytics.total_essays}</div>
                        <div class="stat-label">Total Essays</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${analytics.score_statistics.mean.toFixed(1)}</div>
                        <div class="stat-label">Average Score</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${analytics.word_count_statistics.mean.toFixed(0)}</div>
                        <div class="stat-label">Avg Word Count</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-value">${Object.keys(analytics.grade_level_distribution).length}</div>
                        <div class="stat-label">Grade Levels</div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="chart-container">
                        <h6>Score Distribution</h6>
                        <canvas id="scoreChart"></canvas>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="chart-container">
                        <h6>Grade Level Distribution</h6>
                        <canvas id="gradeChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="chart-container">
                        <h6>Recent Essays</h6>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Title</th>
                                        <th>Score</th>
                                        <th>Grade Level</th>
                                        <th>Word Count</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${analytics.recent_essays.map(essay => `
                                        <tr>
                                            <td>${this.escapeHtml(essay.title)}</td>
                                            <td><span class="essay-score ${this.getScoreClass(essay.score)}">${essay.score.toFixed(1)}</span></td>
                                            <td>${essay.grade_level || 'N/A'}</td>
                                            <td>${essay.word_count}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Create charts
        this.createScoreChart(analytics);
        this.createGradeChart(analytics);
    }

    createScoreChart(analytics) {
        const ctx = document.getElementById('scoreChart').getContext('2d');
        
        // Create score ranges
        const ranges = ['0-2', '2-4', '4-6', '6-8', '8-10'];
        const counts = [0, 0, 0, 0, 0];
        
        analytics.recent_essays.forEach(essay => {
            const score = essay.score;
            if (score < 2) counts[0]++;
            else if (score < 4) counts[1]++;
            else if (score < 6) counts[2]++;
            else if (score < 8) counts[3]++;
            else counts[4]++;
        });

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ranges,
                datasets: [{
                    label: 'Number of Essays',
                    data: counts,
                    backgroundColor: [
                        '#dc3545',
                        '#fd7e14',
                        '#ffc107',
                        '#17a2b8',
                        '#28a745'
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createGradeChart(analytics) {
        const ctx = document.getElementById('gradeChart').getContext('2d');
        
        const labels = Object.keys(analytics.grade_level_distribution);
        const data = Object.values(analytics.grade_level_distribution);

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#007bff',
                        '#28a745',
                        '#ffc107',
                        '#dc3545',
                        '#6f42c1'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Model Training
    async trainModel() {
        try {
            this.showLoading('training-status', 'Training model...');

            const response = await fetch('/api/train', {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayTrainingResults(result);

        } catch (error) {
            console.error('Error training model:', error);
            this.showAlert('Failed to train model: ' + error.message, 'danger');
            this.hideLoading('training-status');
        }
    }

    displayTrainingResults(result) {
        const container = document.getElementById('training-status');
        
        container.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                ${result.message}
            </div>
            
            ${result.metrics ? `
                <div class="mt-3">
                    <h6>Model Performance</h6>
                    <div class="row">
                        <div class="col-6">
                            <div class="feature-item">
                                <div class="feature-value">${result.metrics.mse.toFixed(3)}</div>
                                <div class="feature-label">MSE</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="feature-item">
                                <div class="feature-value">${result.metrics.mae.toFixed(3)}</div>
                                <div class="feature-label">MAE</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="feature-item">
                                <div class="feature-value">${result.metrics.r2.toFixed(3)}</div>
                                <div class="feature-label">R² Score</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="feature-item">
                                <div class="feature-value">${result.metrics.mape.toFixed(1)}%</div>
                                <div class="feature-label">MAPE</div>
                            </div>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
    }

    // Utility Functions
    showLoading(containerId, message = 'Loading...') {
        const container = document.getElementById(containerId);
        container.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">${message}</p>
            </div>
        `;
    }

    hideLoading(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    getScoreClass(score) {
        if (score >= 8.5) return 'score-excellent';
        if (score >= 7.0) return 'score-good';
        if (score >= 5.5) return 'score-average';
        return 'score-poor';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global functions for HTML onclick handlers
function showSection(sectionName) {
    app.showSection(sectionName);
}

function showAddEssayForm() {
    app.showAddEssayForm();
}

function searchEssays() {
    const query = document.getElementById('search-essays').value;
    app.searchEssays(query);
}

function trainModel() {
    app.trainModel();
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new EssayScoringApp();
});
