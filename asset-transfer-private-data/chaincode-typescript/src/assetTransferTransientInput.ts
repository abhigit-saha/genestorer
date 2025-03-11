/*
  SPDX-License-Identifier: Apache-2.0
*/

import { nonEmptyString, positiveNumber } from './utils';

export class TransientAssetProperties {
    objectType: string;
    assetID: string;
    data: string;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('asset_properties');
        if (!transient?.length) {
            throw new Error('no asset properties');
        }
        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(
            json
        ) as Partial<TransientAssetProperties>;

        this.objectType = nonEmptyString(
            properties.objectType,
            'objectType field must be a non-empty string'
        );
        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
        this.data = nonEmptyString(
            properties.data,
            'DNA Data should be non empty'
        );
    }
}

export class TransientAssetValue {
    assetID: string;
    appraisedValue: number;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('asset_value');
        if (!transient?.length) {
            throw new Error('no asset value');
        }
        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(json) as Partial<TransientAssetValue>;

        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
        this.appraisedValue = positiveNumber(
            properties.appraisedValue,
            'appraisedValue field must be a positive integer'
        );
    }
}

export class TransientAssetOwner {
    assetID: string;
    buyerMSP: string;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('asset_owner');
        if (!transient?.length) {
            throw new Error('no asset owner');
        }
        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(json) as Partial<TransientAssetOwner>;

        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
        this.buyerMSP = nonEmptyString(
            properties.buyerMSP,
            'buyerMSP field must be a non-empty string'
        );
    }
}

export class TransientAssetDelete {
    assetID: string;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('asset_delete');
        if (!transient?.length) {
            throw new Error('no asset delete');
        }
        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(json) as Partial<TransientAssetOwner>;

        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
    }
}

export class TransientAssetPurge {
    assetID: string;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('asset_purge');
        if (!transient?.length) {
            throw new Error('no asset purge');
        }

        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(json) as Partial<TransientAssetOwner>;

        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
    }
}

export class TransientAgreementDelete {
    assetID: string;

    constructor(transientMap: Map<string, Uint8Array>) {
        const transient = transientMap.get('agreement_delete');
        if (!transient?.length) {
            throw new Error('no agreement delete');
        }

        const json = Buffer.from(transient).toString();
        const properties = JSON.parse(json) as Partial<TransientAssetOwner>;

        this.assetID = nonEmptyString(
            properties.assetID,
            'assetID field must be a non-empty string'
        );
    }
}
