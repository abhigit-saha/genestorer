/*
  SPDX-License-Identifier: Apache-2.0
*/

import { Object, Property } from 'fabric-contract-api';
import { nonEmptyString, positiveNumber } from './utils';

@Object()
// AssetPrivateDetails describes details that are private to owners
export class AssetPrivateDetails {
    @Property()
    ID: string = '';
    @Property()
    data: string = '';

    static fromBytes(bytes: Uint8Array): AssetPrivateDetails {
        if (bytes.length === 0) {
            throw new Error('no asset private details');
        }
        const json = Buffer.from(bytes).toString();
        const properties = JSON.parse(json) as Partial<AssetPrivateDetails>;

        const result = new AssetPrivateDetails();
        result.ID = nonEmptyString(
            properties.ID,
            'ID field must be a non-empty string'
        );
        result.data = nonEmptyString(
            properties.data,
            'DNA Data should be non empty'
        );

        return result;
    }
}
