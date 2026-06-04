<template>
    <div class="container">
        <h1 class="pt-5 pb-3">
            Reporting Obligation Extraction
        </h1>

        <div class="alert alert-warning alert-dismissible fade show" role="alert">
            <strong>Quick instructions</strong><br/>
            Insert the block of text you wish to find reporting obligations in. 
            Select a model from the dropdown menu to be used for the classification process.
            Press "Begin Extraction" to start the inference process.
            You will see the entire text with the located reporting obligations highlighted.
            Optionally, you can choose a sentence split to only show the extracted sentences.
            All labels over 0.01 in confidence are shown, but those having that value under the selected
            threshold are greyed out.

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
                        <label for="title_field" class="form-label">Insert title:</label>
                        <input id="title_field" class="form-control" type="text" v-model.trim="title"
                               :disabled="random_chosen">
                    </div>
                </div>
                <div class="row mb-3" v-if="use_simple">
                    <div class="col-md-12">
                        <label for="text_field" class="form-label">Insert text:</label>
                        <textarea id="text_field" rows="5" class="form-control" v-model.trim="text"
                                  :disabled="random_chosen"></textarea>
                    </div>
                </div>
                <div class="row mb-3" v-if="random_chosen">
                    <div class="col-12">
                        <ul class="list-group">
                            <li class="list-group-item" v-for="(result, index) in gold_results" :key="index">
                                <span v-if="result.label in results_ok" class="me-3">
                                    <i class="text-success bi bi-arrow-right-circle"
                                       v-if="results_ok[result.label]"></i>
                                    <i class="text-danger bi bi-arrow-right-circle" v-else></i>
                                </span>
                                <span class="badge text-bg-info">GOLD</span>
                                <span class="badge text-bg-success ms-2">euvoc-{{ result.label }}</span>
                                {{ result.description }}
                                <span v-if="result.mapping" class="badge text-bg-warning ms-2">→ euvoc-{{
                                        result.mapping.label
                                    }} - {{ result.mapping.description }}</span>
                                <span v-if="result.do" class="badge text-bg-warning ms-2">→ euvoc-{{
                                        result.do.label
                                    }} - {{ result.do.description }}</span>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="row form-group">
                    <div class="mb-3 col-12 col-md-2" v-if="use_simple">
                        <button class="form-control btn btn-info" @click.prevent="pickRandom" :disabled="random_chosen">
                            Pick random
                        </button>
                    </div>
                    <div class="mb-3 col-12 col-md-2" v-if="use_simple">
                        <button class="form-control btn btn-warning" @click.prevent="clear">Clear form</button>
                    </div>
                    <label class="mb-3 col-6 col-md-1 col-xl-1 col-form-label text-end">
                        Model:
                    </label>
                    <div class="mb-3 col-6 col-md-4 col-xl-3">
                        <select class="form-select" v-model="model" :disabled="random_chosen">
                            <option selected="selected" value="">[Select]</option>
                            <option v-for="(t, index) in models" :key="index" :value="index">
                                {{ ucFirst(t) }}
                            </option>
                        </select>
                    </div>
                    <div class="mb-3 col-12 col-md-3 col-xl-2 ps-5 text-center" v-if="!use_simple">
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="checkbox" id="greedy" value="greedy" v-model="greedy"
                                   :disabled="!use_max || !use_threshold">
                            <label class="form-check-label" for="greedy">Greedy</label>
                            <span class="ms-2" data-bs-toggle="tooltip"
                                  data-bs-title="Stabilisce se le due regole seguenti vanno in AND oppure in OR"><i
                                class="bi bi-info-circle"></i></span>
                        </div>
                        <label class="col-form-label">
                            &nbsp;
                        </label>
                    </div>
                    <div class="mb-3 col-6 col-xl-2 text-end" v-if="!use_simple">
                        <div class="form-check form-check-inline">
                            <input class="form-check-input" type="checkbox" id="max_check" value="max_check"
                                   v-model="use_max">
                            <label class="form-check-label" for="max_check">Use</label>
                        </div>
                        <label class="col-form-label">
                            Max results:
                        </label>
                    </div>
                    <div class="mb-3 col-6 col-xl-1" v-if="!use_simple">
                        <input class="form-control" name="max" v-model="max" type="number" min="1" max="50"
                               :disabled="!use_max"/>
                    </div>
                    <div class="mb-3 col-1 col-xl-2 text-end">
                        <div class="form-check form-check-inline" v-if="!use_simple">
                            <input class="form-check-input" type="checkbox" id="threshold_check" value="threshold_check"
                                   v-model="use_threshold">
                            <label class="form-check-label" for="threshold_check">Use</label>
                        </div>
                        <label class="col-form-label">
                            Threshold:
                        </label>
                    </div>
                    <div class="mb-3 col-2 col-xl-1">
                        <input class="form-control" name="max" v-model="threshold" type="number" min="0.05" max="1.0"
                               step="0.05" :disabled="!use_threshold" @change="recalculateValidation"/>
                    </div>
                </div>
                <div class="row row-cols-lg-auto g-3 align-items-center">
                    <div class="col-12" style="margin-left: auto;">
                        <button class="btn btn-primary px-5 btn-lg" @click.prevent="send">Send data</button>
                    </div>
                </div>
            </form>
            <div v-if="launched">
                <h2 class="py-4">
                    Results
                </h2>
                <div v-if="loading_results">
                    <div class="spinner-border" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <div v-else>
                    <div v-if="results.length == 0">
                        No results to show.
                    </div>
                    <ul v-else class="list-group">
                        <li class="list-group-item" :class="{'greyed': result.score < threshold}"
                            v-for="(result, index) in results" :key="index">
                            <span v-if="random_chosen && result.label in results_ok_r" class="me-3">
<!--                                <i class="text-success bi bi-check-circle" v-if="results_ok_r[result.label] && results_ok[result.label]"></i>-->
                                <!--                                <i class="text-warning bi bi-check-circle" v-else-if="results_ok_r[result.label]"></i>-->
                                <i class="text-success bi bi-check-circle" v-if="results_ok_r[result.label]"></i>
                                <i class="text-danger bi bi-x-circle" v-else></i>
                            </span>
                            <span class="badge text-bg-primary">{{ result.score.toFixed(2) }}</span>
                            <span class="badge text-bg-success ms-2">euvoc-{{ result.label }}</span>
                            {{ result.description }}
                            <span v-if="result.mt" class="badge text-bg-warning ms-2">→ euvoc-{{
                                    result.mt.label
                                }}<template
                                    v-if="result.mt.description"> - {{ result.mt.description }}</template></span>
                            <span v-if="result.do" class="badge text-bg-warning ms-2">→ euvoc-{{
                                    result.do.label
                                }}<template
                                    v-if="result.do.description"> - {{ result.do.description }}</template></span>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</template>

<script>

import axios from "axios";
import {Tooltip} from 'bootstrap'

const apiPath = process.env.VUE_APP_API_URL ? process.env.VUE_APP_API_URL : '';

let start_data = {
    models: {},
    use_threshold: true,
    use_max: true,
    greedy: true,
    title: "",
    text: "",
    threshold: 0.5,
    max: 10,
    model: "",
    loading_results: false,
    loading_page: true,
    launched: false,
    results: [],
    model_type: "",

    use_simple: true,
    random_chosen: false,
    gold_results: [],
    results_ok: {},
    results_ok_r: {}
};

export default {
    name: 'App',
    data() {
        return {...start_data}
    },
    methods: {
        pickRandom: function () {
            let thisBak = this;
            let selectedLang = thisBak.model;
            this.clear();
            axios.get(apiPath + "/api/random", {"params": {"lang": selectedLang}}).then(function (response) {
                // thisBak.models = response.data;
                // console.log(response);
                thisBak.random_chosen = true;
                thisBak.gold_results = response.data.labels;
                thisBak.text = response.data.text;
                thisBak.title = response.data.title;

                thisBak.model = response.data.lang;
            }).catch(function () {
                alert("Error in retrieving data.")
            }).then(function () {
                // thisBak.loading_page = false;
            });
        },
        clear: function () {
            let tmpModels = this.models;
            Object.assign(this.$data, start_data);
            this.models = tmpModels;
            this.loading_page = false;
        },
        ucFirst: function (str) {
            if (str.length > 0) {
                str = str.charAt(0).toUpperCase() + str.slice(1);
            }
            return str;
        },
        send: function () {

            let thisBak = this;

            if (!this.model) {
                alert("Please select model");
                return;
            }

            if (!this.text && !this.title) {
                alert("Please insert a text");
                return;
            }

            this.launched = true;
            this.loading_results = true;
            let requestData = {};
            requestData["title"] = this.title;
            requestData["text"] = this.text;
            requestData['model'] = this.model;
            if (this.use_simple) {
                requestData["threshold"] = this.threshold;
            } else {
                if (this.use_threshold) {
                    // requestData["threshold"] = this.threshold / 100;
                    requestData["threshold"] = this.threshold;
                }
                if (this.use_max) {
                    requestData["top_k"] = this.max;
                }
                if (this.use_max && this.use_threshold) {
                    requestData['greedy'] = this.greedy;
                }
            }
            let post_config = process.env.VUE_APP_TOKEN ? {headers: {"Token": process.env.VUE_APP_TOKEN}} : {};
            axios.post(apiPath + "/api/predict", requestData, post_config).then(function (response) {
                thisBak.results = response.data;
                thisBak.model_type = thisBak.models[thisBak.model];
                thisBak.recalculateValidation();
            }).catch(function (e) {
                alert("Error!");
                console.log(e);
            }).then(function () {
                thisBak.loading_results = false;
            });
        },
        recalculateValidation: function () {
            let thisBak = this;

            if (thisBak.results.length === 0) {
                return
            }

            thisBak.results_ok = {};
            thisBak.results_ok_r = {};

            const gold_labels = new Set();
            for (let o of thisBak.gold_results) {
                gold_labels.add(o.label);
                thisBak.results_ok[o.label] = false;
            }
            for (let o of thisBak.results) {
                if (gold_labels.has(o.label)) {
                    thisBak.results_ok[o.label] = false;
                    thisBak.results_ok_r[o.label] = true;
                    if (o.score >= thisBak.threshold) {
                        thisBak.results_ok[o.label] = true;
                    }
                } else {
                    thisBak.results_ok[o.label] = false;
                    thisBak.results_ok_r[o.label] = false;
                }
            }
        }
    },
    mounted() {
        new Tooltip(document.body, {
            selector: "[data-bs-toggle='tooltip']",
        })
        let thisBak = this;
        axios.get(apiPath + "/api/models").then(function (response) {
            thisBak.models = response.data;
        }).catch(function () {
            alert("Unable to get the list of models, please try again.")
        }).then(function () {
            thisBak.loading_page = false;
        });
    }
}
</script>

<style>
#app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    /*text-align: center;*/
    color: #2c3e50;
    /*margin-top: 60px;*/
}

.list-group-item.greyed {
    opacity: 0.5;
    background-color: #ddd;
}

</style>
