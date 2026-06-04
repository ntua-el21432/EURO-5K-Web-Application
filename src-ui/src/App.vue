<template>
    <div class="container pb-5">
        <h1 class="pt-5 pb-3">
            Reporting Obligation Extraction
        </h1>

        <div class="alert alert-warning alert-dismissible fade show" role="alert">
            <strong>Instructions</strong><br/>
            Insert the text you want to analyze, choose a model,
            then press "Find Reporting Obligations". Sentences detected as obligations will be highlighted.
            Hover over any sentence to see its Confidence score.
            You can also choose to analyze the text in sentence-split format.
            If reporting obligations are detected then you can export to RDF format with the "Export" button. 
            Press "Explain" to generate a LIME/Attention explanation for the selected sentence. This may take a while.
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>

        <div v-if="loading_page">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>

        <div v-else>
            <form>
                <div class="row mb-3">
                    <div class="col-12">
                        <label for="token_field" class="form-label">API Token:</label>
                        <input id="token_field" class="form-control" type="password" v-model.trim="token" placeholder="Enter your AUTH_TOKEN here">
                    </div>
                </div>

                <div class="row mb-3">
                    <label for="text_field" class="form-label">Insert text:</label>
                    <textarea id="text_field" rows="6" class="form-control" v-model.trim="text"></textarea>
                </div>

                <div class="row mb-3">
                    <label class="col-form-label col-12 col-md-2">Model:</label>
                    <div class="col-12 col-md-4">
                        <select class="form-select" v-model="model">
                            <option selected value="">[Select]</option>
                            <option 
                                v-for="(details, key) in models" 
                                :key="key" 
                                :value="key"
                                :disabled="!details.active"
                            >
                                {{ modelDisplayNames[key] || ucFirst(key) }}                           
                            </option>
                        </select>
                    </div>
                </div>

                <div class="row mb-3">
                    <label class="col-form-label col-12 col-md-2">Settings:</label>
                    
                    <div class="col-12 col-md-4 mb-2">
                        <div class="btn-group" role="group">
                            <input type="radio" class="btn-check" name="viewFormat" id="viewInline" value="inline" v-model="viewFormat">
                            <label class="btn btn-outline-secondary" for="viewInline">
                                <i class="bi bi-text-paragraph"></i> Full Text
                            </label>

                            <input type="radio" class="btn-check" name="viewFormat" id="viewList" value="list" v-model="viewFormat">
                            <label class="btn btn-outline-secondary" for="viewList">
                                <i class="bi bi-list-ul"></i> Sentence Split
                            </label>
                        </div>
                    </div>

                    <div class="col-12 col-md-4 d-flex align-items-center">
                        <label class="form-label me-2 mb-0" style="min-width: 100px;">
                            Threshold: <strong>{{ threshold }}</strong>
                        </label>
                        <input 
                            type="range" 
                            class="form-range" 
                            min="0.1" 
                            max="0.99" 
                            step="0.01" 
                            v-model.number="threshold"
                        >
                    </div>
                </div>

                <div class="row mb-3">
                    <div class="col-12">
                        <button class="btn btn-primary px-5 btn-lg" @click.prevent="send">Find Reporting Obligations</button>
                    </div>
                </div>
            </form>

            <div v-if="launched">
                <h2 class="py-4">Results</h2>

                <div v-if="loading_results">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>

                <div v-else>
                    <div v-if="results.length == 0">
                        No results returned.
                    </div>

                    <div v-else>
                        <div class="mb-3 d-flex justify-content-between align-items-center">
                            <span class="badge bg-primary rounded-pill">
                                {{ obligation_count }} Obligations Found
                            </span>
                            
                            <button 
                                v-if="obligation_count > 0"
                                class="btn btn-success btn-sm" 
                                @click="exportRDF"
                                :disabled="exporting"
                            >
                                <i class="bi bi-file-earmark-code"></i> 
                                {{ exporting ? 'Generating...' : 'Export RDF/Turtle' }}
                            </button>
                        </div>

                        <div 
                            class="results-container" 
                            :class="[viewFormat === 'list' ? 'list-view' : 'p-4 border rounded bg-light']"
                        >
                            <span 
                                v-for="(item, index) in results" 
                                :key="index"
                                class="sentence-span"
                                :class="{'highlight-obligation': item.is_reporting_obligation}"
                                data-bs-toggle="tooltip"
                                data-bs-placement="top"
                                :title="'Confidence: ' + (item.score * 100).toFixed(2) + '%'"
                            >
                                <span class="sent-index" v-if="viewFormat === 'list'">#{{index + 1}}</span>
                                {{ item.text }}
                                
                                <button 
                                    class="btn btn-sm ms-2" 
                                    :class="viewFormat === 'list' ? 'btn-outline-primary' : 'btn-dark'"
                                    style="font-size: 0.65em; padding: 1px 5px; vertical-align: middle;"
                                    @click.stop="getExplanation(item.text)"
                                    title="Generate explanation"
                                >
                                    Explain
                                </button>
                            </span>
                        </div>
                        
                        <div class="mt-3 text-muted small">
                            * Hover over text to see confidence score. Click "Explain" for LIME/Attention explanation.
                        </div>
                    </div>
                </div>
                
                <div v-if="loading_explanation" class="mt-5 text-center">
                    <div class="spinner-border text-info" role="status">
                        <span class="visually-hidden">Loading Explanation...</span>
                    </div>
                    <div class="mt-2 text-muted">Generating LIME Explanation...(approx. 4-5 minutes)</div>
                </div>

                <div v-if="explanationHtml" class="mt-5" ref="explanationRef">
                    <div class="card shadow-sm">
                        <div class="card-header d-flex justify-content-between align-items-center bg-white">
                            <h4 class="mb-0">Explanation Analysis</h4>
                            <button class="btn btn-close" @click="explanationHtml = null" aria-label="Close explanation"></button>
                        </div>
                        <div class="card-body overflow-auto">
                            <div v-html="explanationHtml"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
import axios from "axios";
import { Tooltip } from 'bootstrap';

const apiPath = process.env.VUE_APP_API_URL || '';

export default {
    name: 'App',
    data() {
        return {
            models: {},
            text: "",
            threshold: 0.5,
            model: "",
            viewFormat: "inline", 
            loading_results: false,
            loading_page: true,
            launched: false,
            results: [],
            obligation_count: 0,
            explanationHtml: null,
            loading_explanation: false,
            exporting: false,
            token: "",
            modelDisplayNames: {
                'bert_base_fft': 'BERT-base (FFT)',
                'bert_base_lora': 'BERT-base (LoRA)',
                'bert_eurlex_fft': 'Legal-BERT (FFT)',
                'bert_eurlex_lora': 'Legal-BERT (LoRA)',
                'llama': 'Llama-QLoRA',
                'mistral': 'Mistral-QLoRA',
                'saul': 'Saul-QLoRA'
            }
        };
    },
    watch: {
        viewFormat() {
            if (this.text && this.model && this.launched) {
                this.send();
            }
        }
    },
    methods: {
        ucFirst(str) {
            return str.length > 0 ? str.charAt(0).toUpperCase() + str.slice(1) : str;
        },
        send() {
            if (!this.model) {
                alert("Please select a model.");
                return;
            }
            if (!this.text) {
                alert("Please insert a text.");
                return;
            }
            this.launched = true;
            this.loading_results = true;
            this.results = [];
            this.obligation_count = 0;
            this.explanationHtml = null;
            
            let shouldSplit = this.viewFormat === 'list';
            
            let requestData = {
                text: this.text,
                model: this.model,
                threshold: Number(this.threshold), 
                shouldSplitSent: shouldSplit
            };
            
            // prioritize token from this.token, else use .env 
            let post_config = { headers: {} };
            if (this.token) {
                post_config.headers["Token"] = this.token;
            } else if (process.env.VUE_APP_TOKEN) {
                post_config.headers["Token"] = process.env.VUE_APP_TOKEN;
            }

            axios.post(apiPath + "/api/predict", requestData, post_config)
                .then(res => { 
                    this.results = res.data.predictions;
                    this.obligation_count = res.data.obligation_count;
                    
                    // Initializing Tooltips for the new results
                    this.$nextTick(() => {
                        // Dispose of old tooltips to prevent leaks/errors
                        const oldTooltips = document.querySelectorAll('.tooltip');
                        oldTooltips.forEach(el => el.remove());

                        // Select all elements with data-bs-toggle="tooltip"
                        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
                        tooltipTriggerList.map(function (tooltipTriggerEl) {
                            return new Tooltip(tooltipTriggerEl)
                        });
                    });
                })
                .catch(err => { console.error(err); alert("Error connecting to API."); })
                .finally(() => { this.loading_results = false });
        },
        getExplanation(sentenceText) {
            this.loading_explanation = true;
            this.explanationHtml = null;
            let requestData = {
                text: sentenceText,
                model: this.model
            };

            let post_config = { headers: {} };
            if (this.token) {
                post_config.headers["Token"] = this.token;
            } else if (process.env.VUE_APP_TOKEN) {
                post_config.headers["Token"] = process.env.VUE_APP_TOKEN;
            }

            axios.post(apiPath + "/api/explain", requestData, post_config)
                .then(res => {
                    this.explanationHtml = res.data.html_content;
                    this.$nextTick(() => {
                        if (this.$refs.explanationRef) {
                            this.$refs.explanationRef.scrollIntoView({ behavior: 'smooth' });
                        }
                    });
                })
                .catch(err => {
                    console.error(err);
                    alert("Failed to generate explanation.");
                })
                .finally(() => {
                    this.loading_explanation = false;
                });
        },
        exportRDF() {
            this.exporting = true;
            let requestData = { predictions: this.results };
            let config = { 
                responseType: 'blob',
                headers: {}
            };
            
            if (this.token) {
                config.headers["Token"] = this.token;
            } else if (process.env.VUE_APP_TOKEN) {
                config.headers["Token"] = process.env.VUE_APP_TOKEN;
            }

            axios.post(apiPath + "/api/export/rdf", requestData, config)
                .then((response) => {
                    const url = window.URL.createObjectURL(new Blob([response.data]));
                    const link = document.createElement('a');
                    link.href = url;
                    link.setAttribute('download', 'obligations_rrmv.ttl');
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                })
                .catch(err => { console.error(err); alert("Failed to export RDF."); })
                .finally(() => { this.exporting = false; });
        }
    },
    mounted() {
        axios.get(apiPath + "/api/models")
            .then(res => this.models = res.data)
            .catch(() => alert("Unable to get models."))
            .finally(() => this.loading_page = false);
    }
}
</script>

<style>
#app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    color: #2c3e50;
}

.sentence-span {
    padding: 2px 0;
    line-height: 1.8;
    margin-right: 4px; 
    cursor: default; 
    transition: all 0.3s;
    border-radius: 3px;
}

.sentence-span:hover {
    background-color: rgba(0,0,0,0.02);
}

.highlight-obligation {
    background-color: transparent; 
    border-bottom: 3px solid #dc3545; 
    padding-bottom: 2px;
    font-weight: 500;
    cursor: help; 
}

.highlight-obligation:hover {
    background-color: rgba(220, 53, 69, 0.05);
}

.sent-index {
    color: #999;
    font-size: 0.8em;
    font-weight: bold;
    margin-right: 10px;
    display: inline-block;
    width: 30px;
}

.results-container.list-view {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.results-container.list-view .sentence-span {
    display: block;        
    margin-bottom: 15px;    
    padding: 20px 25px;     
    background-color: #fff; 
    border: 1px solid #dee2e6;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    border-left: 5px solid #adb5bd; 
    position: relative; 
}

.results-container.list-view .sentence-span:hover {
    box-shadow: 0 8px 15px rgba(0,0,0,0.08);
    transform: translateY(-2px); 
}

.results-container.list-view .highlight-obligation {
    background-color: #fff;  
    border: 1px solid #dee2e6;
    border-left: 5px solid #adb5bd; 
    border-bottom: 4px solid #dc3545 ;
    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.1); 
}

.results-container.list-view button {
    float: right;
    margin-top: -2px;
}
</style>